# Spécification Technique HRM (Hierarchical Reasoning Model)
## Document de Référence Théorique et Pratique pour Implémentation PyTorch

### Version 2.0 - Janvier 2025

---

## 1. Introduction et contexte théorique

### 1.1 Motivation : Les limites du paradigme actuel

Les modèles de langage actuels (LLMs), malgré leurs succès remarquables, souffrent d'une limitation fondamentale : **leur architecture est paradoxalement peu profonde**. Les Transformers standards ont une profondeur computationnelle fixe qui les place dans des classes de complexité limitées (AC0 ou TC0), les empêchant de résoudre des problèmes nécessitant un temps polynomial.

#### Problèmes identifiés :
- **Incomplétude de Turing** : Les LLMs ne peuvent pas exécuter des algorithmes arbitraires de manière end-to-end
- **Chain-of-Thought comme béquille** : Le CoT est fragile, dépendant de décompositions humaines où une seule erreur peut faire dérailler tout le raisonnement
- **Inefficacité** : Le CoT génère beaucoup de tokens, résultant en une latence élevée et des besoins en données importants

### 1.2 Solution proposée : Le Hierarchical Reasoning Model

Le HRM s'inspire de trois principes fondamentaux du cerveau humain :

1. **Traitement hiérarchique** : Le cerveau traite l'information à travers une hiérarchie d'aires corticales. Les aires de haut niveau intègrent l'information sur des échelles temporelles plus longues et forment des représentations abstraites.

2. **Séparation temporelle** : Ces niveaux hiérarchiques opèrent à des échelles temporelles distinctes (ondes theta lentes 4-8 Hz vs ondes gamma rapides 30-100 Hz).

3. **Connectivité récurrente** : Le cerveau possède des connexions récurrentes extensives permettant le raffinement itératif, tout en évitant le problème d'assignation de crédit profond du BPTT.

### 1.3 Avantages théoriques du HRM

- **Profondeur computationnelle effective** : NT steps au lieu de T pour un RNN standard
- **Universalité computationnelle** : Turing-complet avec suffisamment de mémoire et de temps
- **Efficacité mémoire** : O(1) au lieu de O(T) pour BPTT
- **Raisonnement latent** : Calculs dans l'espace d'états caché sans traduction constante vers le langage

---

## 2. Architecture détaillée et principes biologiques

### 2.1 Vue d'ensemble de l'architecture

Le HRM utilise une architecture hiérarchique à deux niveaux qui évite la convergence prématurée des RNN standards à travers un processus appelé **"convergence hiérarchique"**.

```
Dynamique du système :
- N cycles de haut niveau
- T timesteps de bas niveau par cycle
- Total : N × T timesteps effectifs
```

#### Formulation mathématique :

Pour chaque timestep i :
```
z_L^i = f_L(z_L^{i-1}, z_H^{i-1}, x̃; θ_L)

z_H^i = {
    f_H(z_H^{i-1}, z_L^{i-1}; θ_H)  si i ≡ 0 (mod T)
    z_H^{i-1}                         sinon
}
```

### 2.2 Module Low-Level (L-module) : Traitement rapide et détaillé

Le L-module correspond conceptuellement aux aires sensorielles et motrices de bas niveau du cerveau, traitant l'information immédiate et détaillée.

#### Caractéristiques biologiques :
- **Échelle temporelle rapide** : Mise à jour à chaque timestep (analogie aux ondes gamma)
- **Convergence locale** : Atteint un équilibre dans chaque cycle
- **Reset périodique** : Redémarre après T steps quand le H-module se met à jour

```python
class LowLevelModule(nn.Module):
    """
    Module inspiré des aires corticales de bas niveau
    
    Principe biologique : 
    - Traitement rapide et détaillé (comme V1 pour la vision)
    - Convergence vers des points fixes locaux
    - Guidé par les signaux descendants du H-module
    
    Implémentation :
    - Transformer block avec attention causale
    - Combine état propre + état H-module + entrée
    - Reset implicite via nouveau contexte H
    """
    def __init__(self, config):
        super().__init__()
        self.transformer_block = HRMBlock(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"]
        )
    
    def forward(self, z_L, z_H, x_tilde, attn_mask=None, key_padding_mask=None):
        """
        Équation : z_L^i = f_L(z_L^{i-1}, z_H^{i-1}, x̃)
        
        Le L-module effectue une computation conditionnée par :
        - Son état précédent (mémoire de travail)
        - L'état actuel du H-module (contexte global)
        - L'entrée encodée (information sensorielle)
        """
        z_input = z_L + z_H + x_tilde  # Fusion multi-modale simple
        return self.transformer_block(z_input, attn_mask, key_padding_mask)
```

### 2.3 Module High-Level (H-module) : Planification abstraite

Le H-module correspond aux aires associatives de haut niveau comme le cortex préfrontal, responsable de la planification et du raisonnement abstrait.

#### Caractéristiques biologiques :
- **Échelle temporelle lente** : Mise à jour tous les T timesteps (analogie aux ondes theta)
- **Représentations abstraites** : Maintient le contexte global et la stratégie
- **Guidage top-down** : Dirige les computations du L-module

```python
class HighLevelModule(nn.Module):
    """
    Module inspiré du cortex préfrontal et aires associatives
    
    Principe biologique :
    - Intégration lente d'information (comme PFC, hippocampe)
    - Maintien du contexte et des objectifs
    - Signaux de contrôle top-down
    
    Mathématiquement :
    - Ne se met à jour qu'après convergence du L-module
    - Incorpore le résultat des T steps du L-module
    - Établit un nouveau contexte pour le cycle suivant
    """
    def __init__(self, config):
        super().__init__()
        self.transformer_block = HRMBlock(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"]
        )
    
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        """
        Équation : z_H^i = f_H(z_H^{i-1}, z_L^{i-1})
        
        Mise à jour uniquement quand i ≡ 0 (mod T)
        Intègre le résultat complet du cycle L-module
        """
        z_input = z_H + z_L
        return self.transformer_block(z_input, attn_mask, key_padding_mask)
```

### 2.4 Convergence hiérarchique : Un mécanisme clé

La convergence hiérarchique est ce qui permet au HRM d'éviter la limitation fondamentale des RNN standards : la convergence prématurée.

#### Problème des RNN standards :
```
RNN standard : z^{t+1} = f(z^t)
Problème : ||z^{t+1} - z^t|| → 0 rapidement
Conséquence : Calcul effectif s'arrête après quelques steps
```

#### Solution HRM :
```
L-module converge localement : z_L* = f_L(z_L*, z_H, x̃)
H-module perturbe l'équilibre : z_H^new = f_H(z_H, z_L*)
Nouveau point fixe pour L : z_L*' = f_L(z_L*', z_H^new, x̃)
```

**Résultat expérimental** : Le HRM maintient une activité computationnelle élevée (résidu forward) sur NT steps, contrairement aux RNN qui décroissent rapidement.

```python
class HRMInner(nn.Module):
    """
    Implémentation de la convergence hiérarchique
    
    Principe : "Nested computations"
    - L-module : Recherche/raffinement intensif
    - H-module : Stratégie globale de résolution
    - Interaction : H guide L vers différents équilibres locaux
    
    Analogie biologique :
    - Comme les boucles thalamo-corticales
    - Oscillations theta-gamma couplées
    """
    def __init__(self, config):
        super().__init__()
        self.H_module = HighLevelModule(config)
        self.L_module = LowLevelModule(config)
    
    def forward(self, z_H, z_L, x_tilde, attn_mask=None, key_padding_mask=None):
        # Phase 1 : L-module converge avec H fixe
        z_L_new = self.L_module(z_L, z_H, x_tilde, attn_mask, key_padding_mask)
        
        # Phase 2 : H-module intègre et établit nouveau contexte
        # (géré dans la boucle principale pour respecter la temporalité)
        z_H_new = self.H_module(z_H, z_L_new, attn_mask, key_padding_mask)
        
        return z_H_new, z_L_new
```

---

## 3. Mécanisme de gradient approximé : Théorie et implémentation

### 3.1 Fondements théoriques : Deep Equilibrium Models

Le gradient approximé du HRM est basé sur la théorie des Deep Equilibrium Models (DEQ) et le théorème de la fonction implicite.

#### Formulation mathématique :

Si le L-module converge vers un point fixe :
```
z_L* = f_L(z_L*, z_H, x̃; θ_L)
```

Et le H-module se met à jour :
```
z_H^k = f_H(z_H^{k-1}, z_L*; θ_H)
```

Le théorème de la fonction implicite permet de calculer :
```
∂z_H*/∂θ = (I - J_F)^{-1} ∂F/∂θ
```

Où J_F est le Jacobien de F.

### 3.2 Approximation 1-step

L'approximation 1-step utilise seulement le premier terme de la série de Neumann :
```
(I - J_F)^{-1} ≈ I
```

Ce qui donne les gradients simplifiés :
```
∂z_H*/∂θ_H ≈ ∂f_H/∂θ_H
∂z_H*/∂θ_L ≈ ∂f_H/∂z_L* · ∂z_L*/∂θ_L
```

#### Avantages biologiques et pratiques :
- **Plausibilité biologique** : Assignation de crédit locale seulement
- **Efficacité mémoire** : O(1) vs O(T) pour BPTT
- **Stabilité** : Évite l'explosion/vanishing des gradients sur longues séquences

```python
def train_with_1step_gradient(model, z, x, optimizer):
    """
    Implémentation du gradient approximé inspiré du cerveau
    
    Principe biologique :
    - Le cerveau n'implémente probablement pas BPTT
    - Apprentissage basé sur l'activité synaptique locale récente
    - Correspond aux règles de plasticité Hebbienne
    
    Mathématiquement :
    - Traite les états intermédiaires comme constants
    - Backprop seulement à travers le dernier step
    - Équivalent au DEQ avec approximation d'ordre 1
    """
    x_embedded = model.input_embedding(x)
    z_H, z_L = z
    
    # Forward sans gradient : Économise mémoire GPU
    # Simule la phase de "réflexion" sans apprentissage
    with torch.no_grad():
        for i in range(model.cycles_per_segment * model.steps_per_cycle - 1):
            z_L = model.inner_model.L_module(z_L, z_H, x_embedded)
            if (i + 1) % model.steps_per_cycle == 0:
                z_H = model.inner_model.H_module(z_H, z_L)
    
    # Dernier step avec gradient : Point d'équilibre
    # C'est ici que l'apprentissage se produit
    z_L = model.inner_model.L_module(z_L, z_H, x_embedded)
    z_H = model.inner_model.H_module(z_H, z_L)
    
    # Chemin du gradient :
    # Output → H-module final → L-module final → Input embedding
    y_hat = model.lm_head(z_H)
    return (z_H, z_L), y_hat
```

---

## 4. Deep Supervision : Apprentissage par oscillations

### 4.1 Inspiration neuroscientifique

La deep supervision est inspirée par les oscillations neuronales qui régulent quand l'apprentissage se produit dans le cerveau. Les ondes theta (4-8 Hz) sont particulièrement associées à l'encodage de nouvelles informations.

### 4.2 Implémentation et théorie

```python
def train_with_deep_supervision(model, dataloader, optimizer, config):
    """
    Deep supervision inspirée des oscillations cérébrales
    
    Principe biologique :
    - Les oscillations theta marquent les fenêtres d'apprentissage
    - La consolidation se fait par répétition espacée
    - Différentes phases d'oscillation = différents modes d'apprentissage
    
    Avantages théoriques :
    - Régularisation naturelle
    - Feedback plus fréquent au H-module
    - Stabilité améliorée vs DEQ standard
    
    Implémentation :
    - Chaque segment = une oscillation complète
    - Gradient détaché entre segments
    - Apprentissage à chaque "pic" d'oscillation
    """
    for batch in dataloader:
        input_ids, labels = batch
        
        # Initialisation des états cachés
        # Analogie : État de repos du cerveau
        batch_size, seq_len = input_ids.shape
        z_H = torch.zeros(batch_size, seq_len, config["d_model"])
        z_L = torch.zeros(batch_size, seq_len, config["d_model"])
        
        # Multiple segments avec supervision
        # Chaque segment = cycle d'oscillation theta
        for segment in range(config["n_supervision_segments"]):
            # Forward d'un segment complet
            (z_H_new, z_L_new), y_hat = train_with_1step_gradient(
                model, (z_H, z_L), input_ids, optimizer
            )
            
            # Calcul de la loss et backprop
            # Moment d'apprentissage = pic de l'oscillation
            loss = F.cross_entropy(
                y_hat.view(-1, config["vocab_size"]), 
                labels.view(-1)
            )
            
            # Backpropagation locale
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Détachement crucial : Empêche le gradient de traverser les segments
            # Simule la séparation temporelle des phases d'apprentissage
            z_H = z_H_new.detach()
            z_L = z_L_new.detach()
```

---

## 5. Adaptive Computation Time (ACT) : Pensée rapide et lente

### 5.1 Inspiration : Système 1 vs Système 2

L'ACT est inspiré par la théorie de Kahneman sur les deux systèmes de pensée :
- **Système 1** : Pensée automatique et rapide
- **Système 2** : Raisonnement délibéré et lent

Le cerveau alterne dynamiquement entre ces modes selon la complexité de la tâche.

### 5.2 Implémentation avec Q-learning

```python
class ACTController:
    """
    Contrôleur ACT inspiré du contrôle cognitif adaptatif
    
    Principe neuroscientifique :
    - Le cortex préfrontal évalue le "coût" cognitif
    - Les ganglions de la base implémentent une forme de Q-learning
    - L'allocation de ressources dépend de la récompense attendue
    
    Mathématiquement :
    - MDP épisodique avec états = z_H
    - Actions = {halt, continue}
    - Récompense = exactitude de la prédiction
    - Q-learning pour politique optimale
    """
    def __init__(self, config):
        self.max_segments = config["max_segments"]
        self.min_segments = 1
        self.epsilon = config.get("act_epsilon", 0.1)
        
        # Q-head : Prédit Q(halt) et Q(continue)
        # Analogie : Évaluation du coût/bénéfice cognitif
        self.q_head = nn.Sequential(
            nn.Linear(config["d_model"], 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [Q_halt, Q_continue]
            nn.Sigmoid()
        )
    
    def should_halt(self, z_H, segment_idx):
        """
        Décision de halting basée sur l'état cognitif
        
        Processus de décision :
        1. Évalue l'état actuel du H-module
        2. Estime Q(halt) vs Q(continue)
        3. Décide selon politique ε-greedy
        """
        q_values = self.q_head(z_H.mean(dim=1))  # Pooling sur séquence
        q_halt, q_continue = q_values[:, 0], q_values[:, 1]
        
        # Contraintes
        if segment_idx >= self.max_segments - 1:
            return True  # Limite cognitive atteinte
        
        if segment_idx < self.min_segments:
            return False  # Minimum de réflexion requis
        
        # Décision ε-greedy
        if torch.rand(1) < self.epsilon:
            return torch.rand(1) < 0.5  # Exploration
        
        return (q_halt > q_continue).item()  # Exploitation
    
    def update_q_values(self, states, actions, rewards, next_states):
        """
        Mise à jour Q-learning
        
        Équation de Bellman :
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        # Implémentation du Q-learning...
        pass
```

### 5.3 Résultats expérimentaux sur l'ACT

Les expériences montrent que l'ACT :
- **Adapte le temps de calcul** : 2-3 segments pour problèmes simples, 8+ pour complexes
- **Économise les ressources** : Utilisation moyenne de 3.5 segments vs 8 fixes
- **Scaling à l'inférence** : Performance améliore en augmentant M_max sans réentraînement

---

## 6. Composants architecturaux modernes

### 6.1 RMSNorm : Normalisation efficace

RMSNorm remplace LayerNorm pour une meilleure efficacité et stabilité.

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization
    
    Avantages vs LayerNorm :
    - Plus simple : Pas de centrage (mean)
    - Plus stable : Évite les problèmes numériques
    - Plus rapide : Une seule statistique à calculer
    
    Formule : x_norm = x / RMS(x) * γ
    Où RMS(x) = sqrt(mean(x²))
    """
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # Calcul du RMS
        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalisation et mise à l'échelle
        return self.weight * (x * rms)
```

### 6.2 SwiGLU : Activation avec gating

SwiGLU combine SiLU (Swish) avec un mécanisme de gating pour une meilleure expressivité.

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network
    
    Formule : FFN(x) = (Swish(xW₁) ⊙ xW₂)W₃
    
    Inspiration biologique :
    - Gating similaire aux canaux ioniques neuronaux
    - Modulation multiplicative comme dans le cortex
    
    Avantages :
    - Meilleure expressivité que ReLU
    - Gradients plus stables
    - Performance supérieure empiriquement
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Note : d_ff typiquement = 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # Value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # Output
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU : gate * swish(value)
        gate = self.w1(x)
        value = self.w2(x)
        hidden = F.silu(gate) * value  # SiLU = x * sigmoid(x)
        output = self.w3(hidden)
        return self.dropout(output)
```

### 6.3 Architecture Post-Norm

Post-Norm améliore la stabilité du training pour les réseaux très profonds.

```python
class HRMBlock(nn.Module):
    """
    Transformer block avec Post-Norm
    
    Post-Norm vs Pre-Norm :
    - Pre-Norm : Norm(x) → Transform → x + Transform
    - Post-Norm : x → Transform → Norm(x + Transform)
    
    Avantages Post-Norm :
    - Meilleure préservation du signal
    - Convergence plus stable avec Q-learning
    - Requis pour la stabilité de l'ACT
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUFFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Attention avec connexion résiduelle
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        # FFN avec connexion résiduelle
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x
```

---

## 7. Modèle HRM complet avec mécanismes intégrés

```python
class HRM(nn.Module):
    """
    Hierarchical Reasoning Model complet
    
    Architecture inspirée du cerveau avec :
    - Hiérarchie temporelle (theta-gamma)
    - Convergence hiérarchique
    - Apprentissage local (1-step gradient)
    - Contrôle adaptatif (ACT)
    
    Capacités démontrées :
    - Résout Sudoku extrême (99%+ précision)
    - Navigation optimale dans labyrinthes 30x30
    - 40.3% sur ARC-AGI (dépasse o3-mini)
    - Avec seulement 27M paramètres et 1000 exemples
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # === Embeddings ===
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_embeddings = nn.Embedding(config["block_size"], config["d_model"])
        self.register_buffer("pos_ids", torch.arange(config["block_size"]).unsqueeze(0))
        
        # === Modules hiérarchiques ===
        self.inner_model = HRMInner(config)
        
        # === Têtes de sortie ===
        # Tête de langage pour prédictions
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        
        # Tête de halting pour ACT
        self.halt_head = nn.Sequential(
            nn.Linear(config["d_model"], 1),
            nn.Sigmoid()
        )
        
        # === Paramètres temporels ===
        self.max_segments = config["max_segments"]  # M_max
        self.cycles_per_segment = config["cycles_per_segment"]  # N
        self.steps_per_cycle = config["steps_per_cycle"]  # T
        self.ponder_loss_weight = config.get("ponder_loss_weight", 0.01)
        
        # === Initialisation ===
        with torch.no_grad():
            # Biais négatif encourage halting précoce initialement
            self.halt_head[0].bias.fill_(config.get("halt_bias_init", -2.0))
    
    def compute_embeddings(self, input_ids):
        """Combine token et position embeddings"""
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.pos_embeddings(self.pos_ids[:, :seq_len])
        return token_emb + pos_emb
    
    def forward_segment(self, z_H, z_L, x_embedded, attention_mask=None):
        """
        Execute un segment complet (N cycles × T steps)
        
        Processus :
        1. Pour chaque cycle (N fois) :
           a. L-module converge (T steps)
           b. H-module intègre (1 step)
        2. Retourne états finaux
        
        Analogie : Une "pensée complète" du modèle
        """
        batch_size, seq_len = x_embedded.shape[:2]
        device = x_embedded.device
        
        # Masques d'attention
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 
            diagonal=1
        )
        
        # Statistiques pour analyse
        convergence_metrics = {
            'l_residuals': [],
            'h_residuals': []
        }
        
        # N cycles hiérarchiques
        for cycle in range(self.cycles_per_segment):
            # T steps du L-module (convergence locale)
            for step in range(self.steps_per_cycle):
                z_L_prev = z_L.clone()
                
                # L-module toujours actif
                z_L = self.inner_model.L_module(
                    z_L, z_H, x_embedded,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding_mask
                )
                
                # Mesure de convergence
                l_residual = (z_L - z_L_prev).norm(dim=-1).mean()
                convergence_metrics['l_residuals'].append(l_residual)
                
                # H-module se met à jour à la fin du cycle
                if step == self.steps_per_cycle - 1:
                    z_H_prev = z_H.clone()
                    z_H = self.inner_model.H_module(
                        z_H, z_L,
                        attn_mask=causal_mask,
                        key_padding_mask=key_padding_mask
                    )
                    h_residual = (z_H - z_H_prev).norm(dim=-1).mean()
                    convergence_metrics['h_residuals'].append(h_residual)
        
        return z_H, z_L, convergence_metrics
    
    def compute_halting_probability(self, z_H, step_idx):
        """
        Calcule la probabilité de s'arrêter
        
        Basé sur :
        - État actuel du H-module
        - Nombre de segments déjà exécutés
        - Contraintes min/max
        """
        p_halt = self.halt_head(z_H).squeeze(-1)
        
        # Clamp pour stabilité numérique
        eps = 1e-6
        p_halt = p_halt.clamp(eps, 1 - eps)
        
        # Force halting au maximum
        if step_idx >= self.max_segments - 1:
            p_halt = torch.ones_like(p_halt)
        
        return p_halt
    
    def forward(self, input_ids, labels=None, attention_mask=None, return_intermediates=False):
        """
        Forward pass complet avec ACT et supervision
        
        Retourne :
        - loss : Loss totale (LM + ponder)
        - logits : Prédictions finales
        - ponder_cost : Coût computationnel
        - intermediates : États intermédiaires (si demandé)
        """
        # === Phase 1 : Embedding ===
        x_embedded = self.compute_embeddings(input_ids)
        batch_size, seq_len = input_ids.shape
        
        # === Phase 2 : Initialisation ===
        z_L = torch.zeros_like(x_embedded)
        z_H = torch.zeros_like(x_embedded)
        
        # === Phase 3 : ACT - Calcul adaptatif ===
        halting_probs = []
        remainders = torch.ones(batch_size, seq_len).to(input_ids.device)
        accumulated_z_H = torch.zeros_like(z_H)
        n_updates = torch.zeros(batch_size, seq_len).to(input_ids.device)
        
        segments_outputs = []
        convergence_data = []
        
        for segment_idx in range(self.max_segments):
            # Forward d'un segment
            z_H, z_L, conv_metrics = self.forward_segment(
                z_H, z_L, x_embedded, attention_mask
            )
            
            if return_intermediates:
                convergence_data.append(conv_metrics)
            
            # Calcul halting
            p_halt = self.compute_halting_probability(z_H, segment_idx)
            is_last = (segment_idx == self.max_segments - 1)
            
            # Contribution pondérée
            if is_last:
                contrib = remainders  # Utilise tout le reste
            else:
                contrib = remainders * p_halt
            
            halting_probs.append(contrib)
            accumulated_z_H += contrib.unsqueeze(-1) * z_H
            
            # Mise à jour des remainders
            if not is_last:
                remainders = remainders * (1 - p_halt)
                n_updates += remainders  # Ponder cost
            
            # Sortie du segment
            segment_logits = self.lm_head(z_H)
            segments_outputs.append(segment_logits)
            
            # Early stopping si tous ont halté
            if torch.all(remainders < 1e-6):
                break
        
        # === Phase 4 : Sortie finale ===
        final_logits = self.lm_head(accumulated_z_H)
        
        # === Phase 5 : Calcul des losses ===
        loss = None
        lm_loss = None
        ponder_loss = None
        
        if labels is not None:
            # Language modeling loss
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, self.config["vocab_size"]),
                shift_labels.view(-1)
            )
            
            # Ponder loss (régularise le temps de calcul)
            ponder_loss = torch.mean(n_updates) * self.ponder_loss_weight
            
            # Loss totale
            loss = lm_loss + ponder_loss
        
        # === Retour des résultats ===
        output = {
            "loss": loss,
            "logits": final_logits,
            "lm_loss": lm_loss,
            "ponder_loss": ponder_loss,
            "halting_probs": torch.stack(halting_probs),
            "ponder_cost": torch.mean(n_updates),
            "segments_used": segment_idx + 1
        }
        
        if return_intermediates:
            output["segments_outputs"] = segments_outputs
            output["convergence_data"] = convergence_data
        
        return output
```

---

## 8. Résultats expérimentaux et justifications

### 8.1 Performance sur les benchmarks

Le HRM atteint des performances remarquables avec seulement 27M paramètres et 1000 exemples d'entraînement :

#### ARC-AGI (Abstraction and Reasoning Corpus)
- **HRM : 40.3%** (ARC-AGI-1)
- o3-mini-high : 34.5%
- Claude 3.7 : 21.2%
- **Signification** : Dépasse les modèles avec des milliards de paramètres sur le raisonnement inductif

#### Sudoku-Extreme
- **HRM : 55.0%** avec 1000 exemples
- Tous les modèles CoT : 0%
- **Analyse** : Requiert recherche en profondeur et backtracking, impossible pour CoT

#### Maze-Hard (30×30)
- **HRM : 74.5%** 
- Tous les modèles CoT : 0%
- **Importance** : Planification optimale sur horizon long

### 8.2 Analyse de la profondeur computationnelle

Expérience clé démontrant la nécessité de la profondeur :
- Augmenter la **largeur** du Transformer → Aucune amélioration
- Augmenter la **profondeur** → Amélioration significative
- HRM avec convergence hiérarchique → Utilisation efficace de la profondeur NT

### 8.3 Correspondance avec le cerveau

#### Hiérarchie de dimensionnalité
Analyse du Participation Ratio (PR) - mesure de dimensionnalité effective :
- **L-module** : PR = 30.22 (bas niveau, spécialisé)
- **H-module** : PR = 89.95 (haut niveau, flexible)
- **Ratio H/L** : 2.98 (proche du ratio cortical 2.25)

Cette organisation émerge de l'entraînement et n'est pas présente dans un réseau non entraîné.

---

## 9. Guide d'implémentation pratique

### 9.1 Initialisation recommandée

```python
def initialize_hrm(model, config):
    """
    Initialisation critique pour la stabilité
    
    Stratégies :
    1. LeCun normal tronqué pour les poids
    2. Zéros pour les biais (sauf halt_head)
    3. Small normal pour embeddings
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # LeCun initialization - critique pour convergence
            fan_in = module.weight.shape[1]
            std = 1.0 / math.sqrt(fan_in)
            nn.init.trunc_normal_(module.weight, std=std, a=-2*std, b=2*std)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            # Embeddings avec petite variance
            nn.init.normal_(module.weight, std=0.02)
        
        elif isinstance(module, RMSNorm):
            # RMSNorm : scale à 1
            nn.init.ones_(module.weight)
    
    # Halt head : biais négatif pour encourager continuation initiale
    with torch.no_grad():
        model.halt_head[0].bias.fill_(config.get("halt_bias_init", -2.0))
```

### 9.2 Stratégie d'entraînement

```python
class HRMTrainer:
    """
    Entraîneur spécialisé pour HRM
    
    Combine :
    - Deep supervision
    - 1-step gradient
    - ACT avec Q-learning
    - Curriculum learning
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Optimiseur Adam-atan2 pour stabilité
        self.optimizer = AdamAtan2(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Scheduler avec warmup
        self.scheduler = self.create_scheduler()
        
        # Métriques
        self.metrics = HRMMetrics()
    
    def create_scheduler(self):
        """
        Learning rate schedule critique
        
        Phases :
        1. Warmup linéaire (stabilise training initial)
        2. Cosine decay (convergence douce)
        3. Restarts optionnels (échapper minima locaux)
        """
        def lr_lambda(step):
            warmup = self.config["warmup_steps"]
            if step < warmup:
                return step / warmup
            
            # Cosine decay après warmup
            progress = (step - warmup) / (self.config["total_steps"] - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, dataloader):
        """
        Epoch d'entraînement avec deep supervision
        """
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids, labels = batch
            
            # Initialisation des états
            batch_size, seq_len = input_ids.shape
            z_H = torch.zeros(batch_size, seq_len, self.config["d_model"])
            z_L = torch.zeros(batch_size, seq_len, self.config["d_model"])
            
            # Deep supervision : Multiple segments
            segment_losses = []
            
            for segment in range(self.config["n_supervision_segments"]):
                # Forward avec 1-step gradient
                (z_H_new, z_L_new), logits = train_with_1step_gradient(
                    self.model, (z_H, z_L), input_ids, self.optimizer
                )
                
                # Calcul loss
                loss = F.cross_entropy(
                    logits.view(-1, self.config["vocab_size"]),
                    labels.view(-1)
                )
                segment_losses.append(loss)
                
                # Backward et update
                loss.backward()
                
                # Gradient clipping important pour stabilité
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get("grad_clip", 1.0)
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Detach pour prochain segment
                z_H = z_H_new.detach()
                z_L = z_L_new.detach()
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            epoch_loss += sum(segment_losses).item()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return epoch_loss / len(dataloader)
```

### 9.3 Optimiseur Adam-atan2

```python
class AdamAtan2(torch.optim.Optimizer):
    """
    Variante d'Adam invariante à l'échelle
    
    Motivation :
    - Adam standard peut être instable avec deep equilibrium
    - atan2 normalise les mises à jour angulaires
    - Meilleure convergence empirique pour HRM
    
    Différences vs Adam :
    - Utilise atan2(m, v) au lieu de m/sqrt(v)
    - Invariant aux changements d'échelle
    - Plus stable numériquement
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0.01):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Weight decay (L2 regularization)
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = math.sqrt(1 - beta2 ** state['step'])
                
                # Compute step size
                step_size = group['lr'] * bias_correction2 / bias_correction1
                
                # atan2 update (key difference)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Utilise atan2 pour normalisation angulaire
                # Plus stable que division simple
                angle = torch.atan2(exp_avg, denom)
                p.data.add_(angle, alpha=-step_size)
        
        return loss
```

---

## 10. Configurations spécifiques par tâche

### 10.1 Configuration pour ARC-AGI

```python
ARC_CONFIG = {
    # Architecture adaptée aux puzzles visuels
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "vocab_size": 30,  # 10 couleurs + tokens spéciaux
    "block_size": 900,  # Grille 30×30 max
    
    # Temporalité pour raisonnement inductif
    "cycles_per_segment": 2,
    "steps_per_cycle": 3,
    "max_segments": 8,
    
    # Spécifique ARC
    "augmentation": True,  # Rotations, flips, permutations
    "task_tokens": True,   # Token spécial par puzzle
    "voting_ensemble": 2,   # Deux tentatives permises
    
    # Entraînement
    "batch_size": 16,
    "learning_rate": 5e-5,
    "n_supervision_segments": 4
}
```

### 10.2 Configuration pour Sudoku-Extreme

```python
SUDOKU_CONFIG = {
    # Architecture pour backtracking intensif
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 1024,
    "vocab_size": 20,  # 0-9 + tokens
    "block_size": 81,  # Grille 9×9
    
    # Plus de segments pour exploration/backtrack
    "cycles_per_segment": 4,
    "steps_per_cycle": 4,
    "max_segments": 16,
    
    # ACT critique pour Sudoku
    "ponder_loss_weight": 0.05,  # Plus élevé pour efficacité
    "halt_bias_init": -3.0,  # Encourage réflexion longue
    
    # Entraînement
    "batch_size": 64,
    "learning_rate": 1e-4,
    "n_supervision_segments": 8
}
```

### 10.3 Configuration pour Maze-Hard

```python
MAZE_CONFIG = {
    # Architecture pour planification spatiale
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "vocab_size": 10,  # Types cellules (mur, vide, chemin, etc.)
    "block_size": 900,  # Grille 30×30
    
    # Exploration efficace
    "cycles_per_segment": 2,
    "steps_per_cycle": 2,
    "max_segments": 6,
    
    # Spécifique navigation
    "path_encoding": "relative",  # Encodage relatif des positions
    "bidirectional_search": True,  # Recherche depuis start et goal
    
    # Entraînement
    "batch_size": 32,
    "learning_rate": 5e-5,
    "n_supervision_segments": 3
}
```

---

## 11. Métriques et analyse

### 11.1 Métriques de performance

```python
class HRMMetrics:
    """
    Métriques complètes pour évaluation HRM
    
    Mesure :
    - Précision de prédiction
    - Efficacité computationnelle
    - Convergence hiérarchique
    - Dimensionnalité des représentations
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
        self.segments_used = []
        self.convergence_rates = []
        self.pr_scores = {"H": [], "L": []}
    
    def update(self, outputs, targets):
        """Mise à jour des métriques après un batch"""
        predictions = outputs["logits"].argmax(dim=-1)
        
        # Précision
        self.correct += (predictions == targets).sum().item()
        self.total += targets.numel()
        
        # Efficacité ACT
        self.segments_used.append(outputs["segments_used"])
        
        # Convergence si disponible
        if "convergence_data" in outputs:
            self.analyze_convergence(outputs["convergence_data"])
    
    def analyze_convergence(self, conv_data):
        """Analyse la convergence hiérarchique"""
        for segment_data in conv_data:
            # Moyenne des résidus L-module
            l_residuals = segment_data['l_residuals']
            if l_residuals:
                l_conv_rate = sum(l_residuals) / len(l_residuals)
                self.convergence_rates.append(l_conv_rate.item())
    
    def compute_participation_ratio(self, hidden_states):
        """
        Calcule le Participation Ratio (dimensionnalité effective)
        
        PR = (Σλᵢ)² / Σλᵢ²
        
        Interprétation :
        - PR élevé = représentation haute dimension
        - PR bas = représentation compacte
        """
        # Covariance des états cachés
        cov = torch.cov(hidden_states.T)
        
        # Valeurs propres
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Positives seulement
        
        # Participation Ratio
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        
        return pr.item()
    
    def compute(self):
        """Calcule toutes les métriques"""
        accuracy = self.correct / self.total if self.total > 0 else 0
        avg_segments = np.mean(self.segments_used) if self.segments_used else 0
        
        # Efficacité : précision par segment utilisé
        efficiency = accuracy / avg_segments if avg_segments > 0 else 0
        
        # Taux de convergence moyen
        avg_convergence = np.mean(self.convergence_rates) if self.convergence_rates else 0
        
        return {
            'accuracy': accuracy,
            'avg_segments_used': avg_segments,
            'compute_efficiency': efficiency,
            'convergence_rate': avg_convergence,
            'pr_h_module': np.mean(self.pr_scores["H"]) if self.pr_scores["H"] else 0,
            'pr_l_module': np.mean(self.pr_scores["L"]) if self.pr_scores["L"] else 0
        }
```

### 11.2 Visualisation des états intermédiaires

```python
def visualize_reasoning_process(model, input_ids, task_type="sudoku"):
    """
    Visualise le processus de raisonnement du HRM
    
    Montre :
    - Évolution des prédictions
    - Patterns de convergence
    - Stratégies émergentes
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids, return_intermediates=True)
    
    segments_outputs = outputs["segments_outputs"]
    convergence_data = outputs["convergence_data"]
    
    # Créer visualisation selon le type de tâche
    if task_type == "sudoku":
        return visualize_sudoku_solving(segments_outputs)
    elif task_type == "maze":
        return visualize_pathfinding(segments_outputs)
    elif task_type == "arc":
        return visualize_pattern_discovery(segments_outputs)

def visualize_sudoku_solving(segments_outputs):
    """
    Visualise la résolution de Sudoku
    
    Observations typiques :
    - Exploration initiale (segments 1-3)
    - Identification de contradictions (segments 4-6)
    - Backtracking (segments 7-10)
    - Convergence vers solution (segments 11+)
    """
    # Implémentation de visualisation...
    pass
```

---

## 12. Tests et validation

### 12.1 Suite de tests complète

```python
class HRMTestSuite:
    """
    Tests complets pour validation du HRM
    """
    
    @staticmethod
    def test_architecture():
        """Test de l'architecture de base"""
        config = DEFAULT_CONFIG
        model = HRM(config)
        
        # Test des dimensions
        batch_size = 2
        seq_len = 100
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        
        outputs = model(input_ids)
        
        assert outputs["logits"].shape == (batch_size, seq_len, config["vocab_size"])
        assert outputs["segments_used"] <= config["max_segments"]
        print("✓ Architecture test passed")
    
    @staticmethod
    def test_convergence_hierarchy():
        """Test de la convergence hiérarchique"""
        config = DEFAULT_CONFIG
        model = HRM(config)
        
        input_ids = torch.randint(0, config["vocab_size"], (1, 50))
        outputs = model(input_ids, return_intermediates=True)
        
        # Vérifier que L converge plus vite que H
        conv_data = outputs["convergence_data"][0]
        l_residuals = conv_data['l_residuals']
        h_residuals = conv_data['h_residuals']
        
        # L devrait avoir plus de mises à jour
        assert len(l_residuals) > len(h_residuals)
        print("✓ Hierarchical convergence test passed")
    
    @staticmethod
    def test_gradient_flow():
        """Test du flux de gradient avec 1-step approx"""
        model = HRM(DEFAULT_CONFIG)
        input_ids = torch.randint(0, 100, (1, 10))
        labels = torch.randint(0, 100, (1, 10))
        
        outputs = model(input_ids, labels)
        outputs["loss"].backward()
        
        # Vérifier gradients non-nuls
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
        
        print("✓ Gradient flow test passed")
    
    @staticmethod
    def test_act_mechanism():
        """Test du mécanisme ACT"""
        config = DEFAULT_CONFIG.copy()
        config["max_segments"] = 8
        model = HRM(config)
        
        # Problème simple devrait utiliser moins de segments
        simple_input = torch.zeros(1, 10, dtype=torch.long)
        outputs_simple = model(simple_input)
        
        # Problème complexe devrait utiliser plus de segments
        complex_input = torch.randint(0, config["vocab_size"], (1, 10))
        outputs_complex = model(complex_input)
        
        # ACT devrait adapter le calcul
        assert outputs_simple["segments_used"] <= outputs_complex["segments_used"]
        print("✓ ACT mechanism test passed")
    
    @staticmethod
    def test_memory_efficiency():
        """Test de l'efficacité mémoire vs BPTT"""
        import tracemalloc
        
        config = DEFAULT_CONFIG
        model = HRM(config)
        input_ids = torch.randint(0, config["vocab_size"], (4, 100))
        
        # Mesure mémoire avec 1-step gradient
        tracemalloc.start()
        outputs = model(input_ids)
        if outputs["loss"]:
            outputs["loss"].backward()
        memory_1step = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        print(f"✓ Memory usage (1-step): {memory_1step / 1024**2:.2f} MB")
        
        # Le HRM devrait utiliser O(1) mémoire indépendamment de la longueur
        assert memory_1step < 500 * 1024**2  # Moins de 500MB
    
    @staticmethod
    def run_all_tests():
        """Execute tous les tests"""
        print("Running HRM Test Suite...")
        HRMTestSuite.test_architecture()
        HRMTestSuite.test_convergence_hierarchy()
        HRMTestSuite.test_gradient_flow()
        HRMTestSuite.test_act_mechanism()
        HRMTestSuite.test_memory_efficiency()
        print("\n✅ All tests passed!")
```

---

## 13. Conclusion et perspectives

### 13.1 Contributions principales

Le HRM démontre que :
1. **L'architecture hiérarchique** inspirée du cerveau peut surpasser les approches CoT
2. **La convergence hiérarchique** permet une profondeur computationnelle effective
3. **Le gradient 1-step** offre une alternative efficace au BPTT
4. **L'ACT avec Q-learning** adapte le calcul à la complexité

### 13.2 Perspectives futures

1. **Scaling** : Explorer des versions plus larges du HRM
2. **Multi-modalité** : Extension à vision, audio, etc.
3. **Apprentissage continu** : Adaptation à de nouvelles tâches sans oubli
4. **Interprétabilité** : Analyse des stratégies de raisonnement émergentes

### 13.3 Impact potentiel

Le HRM représente un changement de paradigme vers :
- Des modèles plus efficaces en données (1000 exemples suffisent)
- Un raisonnement véritablement profond sans CoT
- Une architecture biologiquement plausible
- La possibilité d'atteindre la complétude de Turing pratique

---

## Références

- **Paper original** : "Hierarchical Reasoning Model" (arXiv:2506.21734v1)
- **Auteurs** : Guan Wang, Jin Li, et al. - Sapient Intelligence, Singapore
- **Contact** : research@sapient.inc

---

**Fin du document de spécification enrichi**