import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import queue
import threading
import random
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import psutil
import gc


class EfficientTokenBatcher:
    """Gère efficacement la création et le regroupement des batches de tokens."""
    
    def __init__(self, tokenizer, max_length=2048, batch_size=8, dynamic_batching=True):
        """
        Args:
            tokenizer: Tokenizer HuggingFace à utiliser
            max_length: Longueur maximale des séquences
            batch_size: Taille des batches à produire
            dynamic_batching: Si True, utilise le batching dynamique pour économiser de la mémoire
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.dynamic_batching = dynamic_batching
        # Pour le batch dynamique, on trie les séquences par longueur
        self.length_sorted_buffer = []
        self.tokenization_stats = {"total_time": 0, "calls": 0, "tokens": 0}
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize un batch de textes efficacement."""
        start_time = time.time()
        
        # Tokenize les textes
        batch = self.tokenizer(
            texts,
            padding='max_length' if not self.dynamic_batching else 'longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Ajoute les labels pour le training
        batch['labels'] = batch['input_ids'].clone()
        
        # Assure que tous les tenseurs sont contigus pour de meilleures performances
        batch = {k: v.contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        # Mise à jour des statistiques
        elapsed = time.time() - start_time
        self.tokenization_stats["total_time"] += elapsed
        self.tokenization_stats["calls"] += 1
        self.tokenization_stats["tokens"] += batch['input_ids'].numel()
        
        return batch
    
    def add_to_length_buffer(self, text: str):
        """Ajoute un texte au buffer trié par longueur."""
        if not self.dynamic_batching:
            return False
            
        # Tokenize le texte sans padding pour connaître sa longueur
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        length = len(tokens)
        
        # Ajoute au buffer
        self.length_sorted_buffer.append((length, text))
        
        # Si le buffer est assez grand, on le trie et on retourne True
        return len(self.length_sorted_buffer) >= self.batch_size * 2
    
    def get_sorted_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Récupère un batch de textes triés par longueur."""
        if not self.dynamic_batching or len(self.length_sorted_buffer) < self.batch_size:
            return None
            
        # Trie par longueur
        self.length_sorted_buffer.sort(key=lambda x: x[0])
        
        # Extrait un batch
        batch_texts = [text for _, text in self.length_sorted_buffer[:self.batch_size]]
        self.length_sorted_buffer = self.length_sorted_buffer[self.batch_size:]
        
        # Tokenize le batch
        return self.tokenize_batch(batch_texts)
    
    def get_random_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Récupère un batch aléatoire."""
        return self.tokenize_batch(texts)
    
    def get_stats(self) -> Dict[str, float]:
        """Récupère les statistiques de tokenization."""
        stats = self.tokenization_stats.copy()
        if stats["calls"] > 0:
            stats["avg_time_per_call"] = stats["total_time"] / stats["calls"]
            stats["tokens_per_second"] = stats["tokens"] / stats["total_time"] if stats["total_time"] > 0 else 0
        return stats


class AccumulationBuffer:
    """Buffer spécialisé pour gradient accumulation, gérant les groupes de batches."""
    
    def __init__(self, capacity: int = 8, accumulation_steps: int = 1, prefetch_factor: int = 2):
        """
        Args:
            capacity: Nombre maximum de groupes de batches à stocker
            accumulation_steps: Nombre d'étapes d'accumulation de gradient
            prefetch_factor: Multiplicateur pour le préchargement
        """
        self.capacity = max(capacity, accumulation_steps * prefetch_factor)
        self.accumulation_steps = accumulation_steps
        self.buffer = queue.Queue(maxsize=self.capacity)
        self.active_group = None
        self.active_index = 0
        self.stats = {"gets": 0, "puts": 0, "waits": 0, "wait_time": 0}
    
    def put(self, batch_group: List[Dict[str, torch.Tensor]], timeout=None) -> bool:
        """Ajoute un groupe de batches au buffer."""
        try:
            start_wait = time.time()
            self.buffer.put(batch_group, block=True, timeout=timeout)
            wait_time = time.time() - start_wait
            
            self.stats["puts"] += 1
            if wait_time > 0.01:  # Considère comme attente si > 10ms
                self.stats["waits"] += 1
                self.stats["wait_time"] += wait_time
                
            return True
        except queue.Full:
            return False
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Récupère le prochain batch, gérant correctement les groupes d'accumulation."""
        # Si on a un groupe actif et qu'il reste des batches
        if self.active_group and self.active_index < len(self.active_group):
            batch = self.active_group[self.active_index]
            self.active_index += 1
            self.stats["gets"] += 1
            return batch
        
        # Récupère un nouveau groupe
        try:
            self.active_group = self.buffer.get(block=True, timeout=30)
            self.active_index = 1
            self.stats["gets"] += 1
            
            # Retourne le premier batch du nouveau groupe
            return self.active_group[0] if self.active_group else None
        except queue.Empty:
            # Timeout atteint
            return None
    
    def qsize(self) -> int:
        """Retourne le nombre de groupes dans le buffer."""
        return self.buffer.qsize()
    
    def empty(self) -> bool:
        """Vérifie si le buffer est vide."""
        return self.buffer.empty() and (self.active_group is None or self.active_index >= len(self.active_group))
    
    def get_stats(self) -> Dict[str, float]:
        """Récupère les statistiques du buffer."""
        stats = self.stats.copy()
        stats["active_index"] = self.active_index if self.active_group else 0
        stats["active_group_size"] = len(self.active_group) if self.active_group else 0
        stats["buffer_size"] = self.buffer.qsize()
        stats["buffer_capacity"] = self.capacity
        return stats


class LLMDataLoader(IterableDataset):
    """
    Data loader optimisé pour l'entraînement de LLM:
    - Préchargement efficace avec threads parallèles
    - Support explicite du gradient accumulation
    - Batching dynamique pour maximiser l'efficacité GPU
    - Monitoring de performance intégré
    """
    
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "CC-MAIN-2024-10",
        split: str = "train",
        tokenizer=None,
        max_length: int = 2048,
        batch_size: int = 8,
        buffer_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 2,
        gradient_accumulation_steps: int = 1,
        start_offset: int = 0,
        dynamic_batching: bool = True,
        prefetch_factor: int = 2,
        report_interval: int = 100,
    ):
        """
        Args:
            dataset_name: Nom du dataset HuggingFace à charger
            dataset_config: Configuration du dataset
            split: Split du dataset à utiliser
            tokenizer: Tokenizer HuggingFace à utiliser
            max_length: Longueur maximale des séquences
            batch_size: Taille des batches
            buffer_size: Taille du buffer (nombre de groupes de batches)
            shuffle: Si True, mélange les exemples
            num_workers: Nombre de threads de préchargement
            gradient_accumulation_steps: Nombre d'étapes d'accumulation de gradient
            start_offset: Index de départ dans le dataset
            dynamic_batching: Si True, utilise le batching dynamique
            prefetch_factor: Facteur de préchargement par rapport à l'accumulation
            report_interval: Intervalle pour rapporter les statistiques
        """
        super().__init__()
        
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.max_length = max_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.start_offset = start_offset
        self.dynamic_batching = dynamic_batching
        self.prefetch_factor = prefetch_factor
        self.report_interval = report_interval
        
        # Initialise le tokenizer si non fourni
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            access_token = os.getenv('HF_TOKEN')
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct", 
                use_fast=True,
                access_token=access_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialise le batcher pour la tokenization efficace
        self.token_batcher = EfficientTokenBatcher(
            self.tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            dynamic_batching=dynamic_batching
        )
        
        # Contrôle des threads et état
        self.should_stop = threading.Event()
        self.workers = []
        self.dataset = None
        
        # Buffer pour les groupes de batches - RÉDUCTION DE LA TAILLE MAXIMALE
        # Prendre jusqu'à 3 fois le nombre de workers, avec un minimum de 2
        # (pas besoin de stocker 64 groupes par défaut, c'est beaucoup trop)
        effective_buffer_size = min(buffer_size, max(2, num_workers * 3))
        
        print(f"Configuring data loader with {num_workers} workers, buffer size: {effective_buffer_size} groups")
        self.batch_buffer = AccumulationBuffer(
            capacity=effective_buffer_size,
            accumulation_steps=gradient_accumulation_steps,
            prefetch_factor=prefetch_factor
        )
        
        # Statistiques de performance
        self.stats = {
            "iterations": 0,
            "batches_processed": 0,
            "examples_processed": 0,
            "start_time": time.time(),
            "batch_times": [],
            "load_failures": 0,
            "tokenize_failures": 0,
            "tokenize_time": 0,
            "worker_queue_full": 0,
            "worker_queue_empty": 0,
            "accumulation_wait_time": 0,
            "accumulation_refills": 0,
            "direct_group_hits": 0,
            "multi_batch_loads": 0, 
        }
        
        # Démarrage immédiat du préchargement
        self._initialize_dataset()
        self._start_workers()
        
        # Attente initiale pour remplir le buffer
        self._wait_for_initial_data()
    
    def _initialize_dataset(self):
        """Initialise le dataset streaming."""
        print(f"Initializing dataset {self.dataset_name}/{self.dataset_config} (split={self.split})")
        try:
            self.dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split=self.split,
                streaming=True
            ).skip(self.start_offset)
            print(f"Dataset initialized successfully")
        except Exception as e:
            print(f"Error initializing dataset: {e}")
            raise
    
    def _prefetch_worker(self, worker_id: int):
        """Fonction exécutée par chaque thread de préchargement."""
        print(f"Worker {worker_id} started")
        try:
            # Chaque worker obtient son propre itérateur du dataset, avec un skip aléatoire
            # pour éviter que tous les workers commencent au même endroit
            random_offset = random.randint(0, 1000) if worker_id > 0 else 0
            try:
                local_dataset = load_dataset(
                    self.dataset_name,
                    name=self.dataset_config,
                    split=self.split,
                    streaming=True
                ).skip(self.start_offset + random_offset)
                dataset_iter = iter(local_dataset)
            except Exception as e:
                print(f"Worker {worker_id} dataset initialization error: {e}")
                return
            
            # Variables pour le traitement par batch
            examples_buffer = []
            pause_time = 0.01  # Temps d'attente initial lorsque le buffer est plein
            max_pause = 1.0    # Temps d'attente maximum
            
            while not self.should_stop.is_set():
                # Vérifier si le buffer est presque plein, dans ce cas ralentir le worker
                buffer_fill_ratio = self.batch_buffer.qsize() / self.batch_buffer.capacity
                
                if buffer_fill_ratio > 0.8:
                    # Le buffer est presque plein, on attend de plus en plus longtemps si ça continue
                    time.sleep(pause_time)
                    pause_time = min(max_pause, pause_time * 1.5)  # Augmentation exponentielle du temps d'attente
                    
                    # Si complètement plein, on attend encore plus
                    if buffer_fill_ratio >= 0.95:
                        time.sleep(0.5)
                        continue
                else:
                    # Le buffer a de la place, on réinitialise le temps d'attente
                    pause_time = 0.01
                
                try:
                    # Phase 1: Remplir le buffer d'exemples
                    while len(examples_buffer) < self.batch_size:
                        # Vérifier régulièrement si on doit s'arrêter ou ralentir
                        if self.should_stop.is_set() or self.batch_buffer.qsize() >= self.batch_buffer.capacity * 0.9:
                            break
                            
                        try:
                            example = next(dataset_iter)
                            examples_buffer.append(example['text'])
                        except Exception as e:
                            # Gestion spécifique des erreurs connues
                            if "Already borrowed" in str(e):
                                # Problème de concurrence, on attend un peu et on continue
                                time.sleep(0.1)
                                continue
                            else:
                                # Autre erreur, on la remonte
                                raise
                        
                    # Si on a assez d'exemples et qu'on utilise le batching dynamique,
                    # on essaie d'abord d'obtenir un batch trié par longueur
                    if self.dynamic_batching and len(examples_buffer) >= self.batch_size:
                        for text in examples_buffer[:self.batch_size]:
                            self.token_batcher.add_to_length_buffer(text)
                        examples_buffer = examples_buffer[self.batch_size:]
                        sorted_batch = self.token_batcher.get_sorted_batch()
                        if sorted_batch:
                            self._add_batch_to_group(sorted_batch, worker_id)
                            continue
                    
                    # Sinon on prend simplement les premiers exemples
                    if len(examples_buffer) >= self.batch_size:
                        batch_texts = examples_buffer[:self.batch_size]
                        examples_buffer = examples_buffer[self.batch_size:]
                        
                        # Mélange si nécessaire
                        if self.shuffle:
                            random.shuffle(batch_texts)
                        
                        # Tokenize le batch
                        batch = self.token_batcher.get_random_batch(batch_texts)
                        
                        # Ajoute au groupe courant
                        added = self._add_batch_to_group(batch, worker_id)
                        if not added:
                            # Si on n'a pas pu ajouter le batch (buffer plein), on attend un peu
                            time.sleep(0.5)
                
                except StopIteration:
                    # Dataset épuisé, on recommence avec un autre itérateur
                    print(f"Worker {worker_id}: dataset iteration complete, restarting...")
                    try:
                        local_dataset = load_dataset(
                            self.dataset_name,
                            name=self.dataset_config,
                            split=self.split,
                            streaming=True
                        ).skip(self.start_offset + random.randint(0, 5000))
                        dataset_iter = iter(local_dataset)
                    except Exception as e:
                        print(f"Worker {worker_id} dataset reset error: {e}")
                        time.sleep(1.0)  # Attendre un peu avant de réessayer
                
                except Exception as e:
                    print(f"Worker {worker_id} error: {e}")
                    # Continue en cas d'erreur sur un batch
                    time.sleep(0.1)
            
            print(f"Worker {worker_id} stopping")
            
        except Exception as e:
            print(f"Fatal error in worker {worker_id}: {e}")
    
    def _add_batch_to_group(self, batch, worker_id):
        """Ajoute un batch au groupe d'accumulation courant."""
        try:
            # Groupe de batches pour accumulation
            with self._accumulation_group_lock:
                # Si le buffer est plein ou presque plein, on refuse d'ajouter plus de batches
                if self.batch_buffer.qsize() >= self.batch_buffer.capacity * 0.9:
                    return False
                    
                # Ajoute au groupe courant
                self._current_group.append(batch)
                
                # Si le groupe est complet, on l'ajoute au buffer
                if len(self._current_group) >= self.gradient_accumulation_steps:
                    # Ajoute au buffer, avec timeout pour éviter les blocages
                    success = self.batch_buffer.put(self._current_group, timeout=1.0)
                    
                    if success:
                        # Réinitialise le groupe
                        self._current_group = []
                        return True
                    else:
                        # Si le buffer est plein, on enlève ce batch du groupe courant
                        # pour éviter la duplication lors d'un prochain essai
                        self._current_group.pop()
                        return False
                
                return True  # Le groupe n'est pas encore complet, mais le batch a été ajouté
        except Exception as e:
            print(f"Error adding batch to group: {e}")
            return False
    
    def _start_workers(self):
        """Démarre les threads de préchargement."""
        # Initialise les structures partagées
        self._current_group = []
        self._accumulation_group_lock = threading.Lock()
        
        # Démarre les workers
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._prefetch_worker,
                args=(i,),
                daemon=True,
                name=f"DataLoader-Worker-{i}"
            )
            worker.start()
            self.workers.append(worker)
    
    def _wait_for_initial_data(self):
        """Attend que le buffer initial soit rempli."""
        print("Waiting for initial data...")
        timeout = 60  # secondes
        start_time = time.time()
        
        # Attente active pour le premier batch avec feedback
        dots = 0
        loading_complete = False
        
        while not loading_complete and time.time() - start_time < timeout:
            # Vérifier si on a un batch complet d'accumulation disponible
            if (self.batch_buffer.active_group and 
                len(self.batch_buffer.active_group) >= self.gradient_accumulation_steps):
                loading_complete = True
                break
                
            # Vérifier si on a des données dans le buffer
            if not self.batch_buffer.empty():
                loading_complete = True
                break
                
            # Attendre un peu et montrer des progrès
            time.sleep(0.5)
            dots = (dots + 1) % 4
            print(f"Waiting for data{'.' * dots}{' ' * (3-dots)}", end="\r")
        
        if not loading_complete:
            print(f"\nWarning: Timed out waiting for initial data after {timeout}s")
        else:
            print(f"\nInitial data loaded in {time.time() - start_time:.2f}s")
            # Diagnostic d'état
            print(f"Buffer size: {self.batch_buffer.qsize()}/{self.batch_buffer.capacity} groups")
            if self.batch_buffer.active_group:
                print(f"Active group size: {len(self.batch_buffer.active_group)} batches")
    
    def __iter__(self):
        """Retourne un itérateur sur le dataset."""
        return self
    
    def __next__(self):
        """Récupère le prochain batch."""
        # Obtient un batch du buffer
        batch = self.batch_buffer.get()
        
        if batch is None:
            # Fin du dataset ou timeout
            self.should_stop.set()
            raise StopIteration
        
        # Mise à jour des statistiques
        self.stats["batches_processed"] += 1
        self.stats["examples_processed"] += self.batch_size
        self.stats["iterations"] += 1
        
        # Calcul du temps par batch pour les 100 derniers batches
        current_time = time.time()
        self.stats["batch_times"].append(current_time)
        if len(self.stats["batch_times"]) > 100:
            self.stats["batch_times"].pop(0)
        
        # Rapport périodique
        if self.stats["iterations"] % self.report_interval == 0:
            self._report_stats()
        
        return batch
    
    def get_accumulation_batches(self, num_steps):
        """
        Récupère d'un coup tous les lots nécessaires pour une étape d'accumulation.
        Cette méthode est plus efficace que d'appeler next() plusieurs fois, car elle évite
        les synchronisations inutiles entre les appels.
        
        Args:
            num_steps: Nombre de micro-steps (batches) à récupérer
            
        Returns:
            List[Dict]: Liste de batches
        """
        # Démarre un timer pour mesurer le temps d'attente global
        start_time = time.time()
        
        # Vérifie si le groupe est disponible directement
        if self.batch_buffer.active_group and len(self.batch_buffer.active_group) >= num_steps:
            # On peut utiliser le groupe actif directement
            batches = self.batch_buffer.active_group[:num_steps]
            self.batch_buffer.active_index = num_steps
            
            # Mise à jour des statistiques
            self.stats["batches_processed"] += num_steps
            self.stats["examples_processed"] += self.batch_size * num_steps
            self.stats["iterations"] += 1
            self.stats["accumulation_refills"] += 1
            self.stats["direct_group_hits"] += 1
            
            # Mesure le temps de récupération
            wait_time = time.time() - start_time
            self.stats["accumulation_wait_time"] += wait_time
            
            # Log plus détaillé au intervalle de rapport
            if self.stats["iterations"] % self.report_interval == 0:
                print(f"⚡ Accumulation fast-path: {wait_time*1000:.1f}ms pour {num_steps} batches (direct hit)")
            
            # Rapport périodique
            if self.stats["iterations"] % self.report_interval == 0:
                self._report_stats()
                
            return batches
        
        # Sinon, on récupère les lots un par un
        batches = []
        get_times = []
        
        for step in range(num_steps):
            step_start = time.time()
            try:
                batch = self.__next__()
                batches.append(batch)
                get_times.append(time.time() - step_start)
            except StopIteration:
                # Si on n'a pas assez de données, on renvoie ce qu'on a
                if not batches:
                    raise StopIteration
                break
        
        # Enregistre le temps total d'attente
        wait_time = time.time() - start_time
        self.stats["accumulation_wait_time"] += wait_time
        self.stats["accumulation_refills"] += 1
        
        # Log plus détaillé
        if self.stats["iterations"] % self.report_interval == 0:
            avg_time = sum(get_times) / len(get_times) if get_times else 0
            print(f"🔄 Accumulation slow-path: {wait_time*1000:.1f}ms total, "
                  f"{avg_time*1000:.1f}ms/batch en moyenne pour {len(batches)}/{num_steps} batches")
        
        return batches
    
    def _report_stats(self):
        """Affiche les statistiques du data loader."""
        # Calcul des statistiques temporelles
        current_time = time.time()
        elapsed = current_time - self.stats["start_time"]
        
        # Calcul des débits récents (sur les 100 derniers batches max)
        recent_times = self.stats["batch_times"]
        if len(recent_times) > 1:
            recent_elapsed = recent_times[-1] - recent_times[0]
            recent_batches = len(recent_times)
            batches_per_sec = recent_batches / recent_elapsed if recent_elapsed > 0 else 0
            examples_per_sec = recent_batches * self.batch_size / recent_elapsed if recent_elapsed > 0 else 0
        else:
            batches_per_sec = 0
            examples_per_sec = 0
        
        # Statistiques globales
        total_batches = self.stats["batches_processed"]
        total_examples = self.stats["examples_processed"]
        avg_batches_per_sec = total_batches / elapsed if elapsed > 0 else 0
        avg_examples_per_sec = total_examples / elapsed if elapsed > 0 else 0
        
        # Statistiques de mémoire
        memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        
        # Statistiques des workers et du buffer
        buffer_stats = self.batch_buffer.get_stats()
        tokens_stats = self.token_batcher.get_stats()
        
        # Statistiques d'accumulation
        accumulation_stats = ""
        if self.stats.get("accumulation_refills", 0) > 0:
            avg_wait = self.stats["accumulation_wait_time"] / self.stats["accumulation_refills"]
            hit_rate = self.stats.get("direct_group_hits", 0) / self.stats["accumulation_refills"] * 100
            accumulation_stats = (
                f"\n🔄 Accumulation: {self.stats['accumulation_refills']} refills, "
                f"{avg_wait:.2f}s avg wait, {hit_rate:.1f}% direct hits"
            )
        
        # Formatage du rapport
        # print(
        #     f"\n===== Data Loader Stats (Iter {self.stats['iterations']}) =====\n"
        #     f"⏱️  Runtime: {elapsed:.2f}s, Memory: {memory_used:.2f}MB\n"
        #     f"📊 Throughput: {batches_per_sec:.2f} batches/s, {examples_per_sec:.2f} examples/s (recent)\n"
        #     f"📈 Average: {avg_batches_per_sec:.2f} batches/s, {avg_examples_per_sec:.2f} examples/s (total)\n"
        #     f"🧠 Buffer: {buffer_stats['buffer_size']}/{buffer_stats['buffer_capacity']} groups, "
        #     f"{buffer_stats['gets']} gets, {buffer_stats['puts']} puts\n"
        #     f"🔤 Tokenizer: {tokens_stats.get('tokens_per_second', 0):.2f} tokens/s, "
        #     f"{tokens_stats.get('tokens', 0)/1e6:.2f}M tokens processed"
        #     f"{accumulation_stats}"
        #     f"\n========================================="
        # )
    
    def __del__(self):
        """Nettoyage à la destruction."""
        print("Shutting down data loader...")
        self.should_stop.set()
        
        # Attente des workers avec timeout
        for i, worker in enumerate(self.workers):
            if worker.is_alive():
                print(f"Waiting for worker {i} to finish...")
                worker.join(timeout=1.0)
                if worker.is_alive():
                    print(f"Worker {i} is still running")
        
        # Aide au garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Data loader shutdown complete")


# Fonction utilitaire pour créer facilement un data loader
def create_llm_data_loader(
    tokenizer=None,
    batch_size=8,
    max_length=2048,
    gradient_accumulation_steps=1,
    dataset_name="HuggingFaceFW/fineweb-edu",
    **kwargs
):
    """
    Crée un LLMDataLoader avec des paramètres raisonnables.
    
    Args:
        tokenizer: Tokenizer HuggingFace à utiliser
        batch_size: Taille des batches
        max_length: Longueur maximale des séquences
        gradient_accumulation_steps: Nombre d'étapes d'accumulation de gradient
        dataset_name: Nom du dataset HuggingFace à charger
        **kwargs: Arguments supplémentaires à passer au LLMDataLoader
    
    Returns:
        Un LLMDataLoader configuré
    """
    # Calcul du buffer_size et du nombre de workers basés sur la configuration
    cpu_count = os.cpu_count() or 4
    recommended_workers = min(4, max(1, cpu_count // 2))
    
    # Calcul du buffer_size plus raisonnable:
    # - Minimum 2 groupes pour éviter les arrêts
    # - Maximum 3 groupes par worker (chaque worker peut préparer 3 groupes)
    # - Pour l'accumulation, on ne garde que le minimum nécessaire + 1 pour éviter les attentes
    recommended_buffer = min(
        max(2, recommended_workers * 2),  # 2 groupes par worker
        max(2, gradient_accumulation_steps + 1)  # Juste le nécessaire pour l'accumulation + 1
    )
    
    # Paramètres par défaut raisonnables
    default_params = {
        "num_workers": kwargs.get("num_workers", recommended_workers),
        "buffer_size": kwargs.get("buffer_size", recommended_buffer),
        "prefetch_factor": kwargs.get("prefetch_factor", 1),  # Réduit à 1 par défaut
        "dynamic_batching": kwargs.get("dynamic_batching", True),
        "report_interval": kwargs.get("report_interval", 10),  # Réduit pour voir les stats plus fréquemment
    }
    
    # Mise à jour avec les kwargs
    params = {**default_params, **kwargs}
    
    print(f"Creating LLMDataLoader with buffer_size={params['buffer_size']}, "
          f"workers={params['num_workers']}, prefetch_factor={params['prefetch_factor']}, "
          f"gradient_accumulation_steps={gradient_accumulation_steps}")
    
    return LLMDataLoader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        **params
    )


# Exemple d'utilisation
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Charger tokenizer (en utilisant un modèle comme exemple)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Créer le data loader
    data_loader = create_llm_data_loader(
        tokenizer=tokenizer,
        batch_size=4,
        max_length=1024,
        gradient_accumulation_steps=2,
    )
    
    # Utilisation dans une boucle d'entraînement
    for i, batch in enumerate(data_loader):
        print(f"Batch {i}: {batch['input_ids'].shape}")
        
        # Arrêt après 10 batches pour cet exemple
        if i >= 10:
            break 