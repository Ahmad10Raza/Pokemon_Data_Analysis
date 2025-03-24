from flask import Flask, render_template, request
import os
from advanced_predictor import AdvancedPokemonPredictor

app = Flask(__name__)

# Initialize predictor
predictor = None

def load_predictor():
    global predictor
    try:
        predictor = AdvancedPokemonPredictor.load(
            filename=os.path.join('models', 'advanced_pokemon_predictor.pkl'),
            data_path='pokemon.csv'
        )
        print("Predictor loaded successfully")
    except Exception as e:
        print(f"Error loading predictor: {str(e)}")
        predictor = None

load_predictor()

def format_stat_differences(features):
    """Helper function to format stat differences"""
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    return {
        'total': sum(features.get(f'{stat}_diff', 0) for stat in stats),
        'attack': features.get('Attack_diff', 0),
        'defense': features.get('Defense_diff', 0),
        'speed': features.get('Speed_diff', 0),
        'hp': features.get('HP_diff', 0),
        'sp_atk': features.get('Sp. Atk_diff', 0),
        'sp_def': features.get('Sp. Def_diff', 0)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    pokemon_list = []
    if predictor and hasattr(predictor, 'data'):
        pokemon_list = predictor.data['Name'].tolist()
    
    if request.method == 'POST':
        pokemon1 = request.form.get('pokemon1', '').strip()
        pokemon2 = request.form.get('pokemon2', '').strip()
        
        if not pokemon1 or not pokemon2:
            return render_template('index.html',
                               error="Please select two Pokémon",
                               pokemon_list=pokemon_list)
        
        if pokemon1 == pokemon2:
            return render_template('index.html',
                               error="Please select two different Pokémon",
                               pokemon_list=pokemon_list)
        
        try:
            result = predictor.predict(pokemon1, pokemon2)
            features = result.get('features', {})
            
            battle_details = {
                'type_advantage': {
                    'pokemon1': f"{features.get('type_advantage_p1', 1.0):.1f}x",
                    'pokemon2': f"{features.get('type_advantage_p2', 1.0):.1f}x"
                },
                'speed': {
                    'attacker': pokemon1 if features.get('speed_tier', 0) == 1 else pokemon2,
                    'value': features.get('Speed_diff', 0)
                },
                'stats': format_stat_differences(features)
            }
            
            return render_template('index.html',
                               pokemon1=pokemon1,
                               pokemon2=pokemon2,
                               winner=result.get('winner'),
                               confidence=f"{result.get('confidence', 0.5)*100:.1f}%",
                               pokemon1_prob=f"{result.get('probabilities', {}).get(pokemon1, 0.5)*100:.1f}%",
                               pokemon2_prob=f"{result.get('probabilities', {}).get(pokemon2, 0.5)*100:.1f}%",
                               battle_details=battle_details,
                               pokemon_list=pokemon_list)
            
        except Exception as e:
            return render_template('index.html',
                               error=f"Prediction error: {str(e)}",
                               pokemon_list=pokemon_list)
    
    return render_template('index.html',
                       pokemon_list=pokemon_list)

if __name__ == '__main__':
    app.run(debug=True)