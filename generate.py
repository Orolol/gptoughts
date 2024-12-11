import os
import torch
import glob
from model import GPTConfig, EncoderDecoderGPT
from transformers import AutoTokenizer
import random
import json
import time

# Configuration
CHECKPOINT_DIR = 'out'
GENERATION_OUTPUT_DIR = 'generations'
NUM_GENERATIONS = 10
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prompt templates
PROMPT_TEMPLATES = [
    "Il était une fois",
    "Dans un futur lointain",
    "Le roi dit",
    "Elle le regarda et",
    "Au fond de la forêt",
    "Quand le soleil se leva",
    "Le vieux sorcier",
    "Dans le château",
    "Le dragon",
    "Au bord de la rivière"
]

def load_model_from_checkpoint(checkpoint_path, device):
    """Charge le modèle à partir d'un checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Récupérer la configuration du modèle
    model_args = checkpoint['model_args']
    
    # Créer le modèle
    model = EncoderDecoderGPT(
        model_args['encoder_config'],
        model_args['decoder_config']
    )
    
    # Charger les poids
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    
    # Convertir le modèle en bfloat16 si disponible, sinon float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
    elif torch.cuda.is_available():
        model = model.to(torch.float16)
    
    model.eval()
    model.to(device)
    
    # Vérifier le dtype du modèle
    model_dtype = next(model.parameters()).dtype
    print(f"Model loaded with dtype: {model_dtype}")
    
    return model, checkpoint['iter_num'], checkpoint['best_val_loss']

def generate_samples(model, tokenizer, device):
    """Génère des échantillons de texte pour chaque prompt"""
    generations = []
    
    # Détecter le dtype du modèle
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")
    
    for prompt in PROMPT_TEMPLATES:
        # Encoder le prompt
        input_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        
        try:
            # Générer le texte avec gestion des types
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=model_dtype):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE
                )
            
            # Convertir le tensor en liste Python avant le décodage
            if isinstance(output_ids, torch.Tensor):
                output_ids = output_ids.cpu().tolist()
            
            # Décoder le texte généré
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generations.append({
                'prompt': prompt,
                'generated': generated_text,
                'output_length': len(output_ids[0])  # Ajouter des métriques utiles
            })
            print(f"Successfully generated text: {prompt} {generated_text}...")
            
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {str(e)}")
            generations.append({
                'prompt': prompt,
                'generated': f"ERROR: {str(e)}",
                'error': str(e)
            })
            continue
    
    return generations

def extract_iter_num(filename):
    """Extrait le numéro d'itération du nom du fichier checkpoint"""
    # Le format est 'ckpt_iter_X_loss_Y.pt'
    try:
        # Prend la partie après 'iter_' et avant '_loss'
        iter_str = filename.split('iter_')[1].split('_loss')[0]
        return int(iter_str)
    except:
        return 0

def main():
    # Créer le dossier de sortie
    os.makedirs(GENERATION_OUTPUT_DIR, exist_ok=True)
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        use_fast=True,
        access_token=os.getenv('HF_TOKEN')
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Trouver tous les checkpoints et les trier par numéro d'itération
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, 'ckpt_iter_*.pt'))
    checkpoints.sort(key=extract_iter_num)
    
    if not checkpoints:
        print(f"No checkpoints found in {CHECKPOINT_DIR}")
        return
        
    print(f"Found {len(checkpoints)} checkpoints")
    
    for ckpt_path in checkpoints:
        print(f"\nProcessing {ckpt_path}")
        # Charger le modèle
        model, iter_num, val_loss = load_model_from_checkpoint(ckpt_path, DEVICE)
        
        # Générer les échantillons
        generations = generate_samples(model, tokenizer, DEVICE)
        
        # Sauvegarder les générations
        output_file = os.path.join(
            GENERATION_OUTPUT_DIR,
            f'generations_iter_{iter_num}_loss_{val_loss:.4f}.json'
        )
        
        # Sauvegarder les générations avec des métadonnées supplémentaires
        output_data = {
            'iteration': int(iter_num),  # Convertir en int pour sûr
            'val_loss': float(val_loss), # Convertir en float pour sûr
            'generations': generations,
            'metadata': {
                'checkpoint_path': ckpt_path,
                'device': str(DEVICE),
                'max_new_tokens': MAX_NEW_TOKENS,
                'temperature': TEMPERATURE,
                'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved generations for iteration {iter_num} to {output_file}")
        
        # Libérer la mémoire
        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main() 