import os
import torch
import glob
from model import GPTConfig, EncoderDecoderGPT
from transformers import AutoTokenizer
import random
import json

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
    model.eval()
    model.to(device)
    
    return model, checkpoint['iter_num'], checkpoint['best_val_loss']

def generate_samples(model, tokenizer, device):
    """Génère des échantillons de texte pour chaque prompt"""
    generations = []
    
    for prompt in PROMPT_TEMPLATES:
        # Encoder le prompt
        input_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        
        # Générer le texte
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )
        
        # Décoder le texte généré
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generations.append({
            'prompt': prompt,
            'generated': generated_text
        })
    
    return generations

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
    
    # Trouver tous les checkpoints
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, 'ckpt_iter_*.pt'))
    checkpoints.sort(key=lambda x: int(x.split('_')[-2]))  # Trier par numéro d'itération
    
    for ckpt_path in checkpoints:
        # Charger le modèle
        model, iter_num, val_loss = load_model_from_checkpoint(ckpt_path, DEVICE)
        
        # Générer les échantillons
        generations = generate_samples(model, tokenizer, DEVICE)
        
        # Sauvegarder les générations
        output_file = os.path.join(
            GENERATION_OUTPUT_DIR,
            f'generations_iter_{iter_num}_loss_{val_loss:.4f}.json'
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'iteration': iter_num,
                    'val_loss': val_loss,
                    'generations': generations
                },
                f,
                ensure_ascii=False,
                indent=2
            )
        
        print(f"Saved generations for iteration {iter_num} to {output_file}")
        
        # Libérer la mémoire
        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main() 