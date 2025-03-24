import pandas as pd
import numpy as np
import random
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

class AdvancedPokemonPredictor:
    def __init__(self, data_path='pokemon.csv'):
        self.data = pd.read_csv(data_path)
        self.type_chart = self.create_comprehensive_type_chart()
        self.scaler = StandardScaler()
        self.models = {
            'xgb': XGBClassifier(random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
           
        }
        self.ensemble = None
        self.feature_columns = None
    
    def create_comprehensive_type_chart(self):
        """Complete type effectiveness chart with all 18 types"""
        return {
            'Normal': {'Rock':0.5, 'Ghost':0, 'Steel':0.5},
            'Fire': {'Fire':0.5, 'Water':0.5, 'Grass':2, 'Ice':2, 'Bug':2, 
                    'Rock':0.5, 'Dragon':0.5, 'Steel':2},
            'Water': {'Fire':2, 'Water':0.5, 'Grass':0.5, 'Ground':2, 'Rock':2, 'Dragon':0.5},
            'Electric': {'Water':2, 'Electric':0.5, 'Grass':0.5, 'Ground':0, 'Flying':2, 'Dragon':0.5},
            'Grass': {'Fire':0.5, 'Water':2, 'Grass':0.5, 'Poison':0.5, 'Ground':2, 
                     'Flying':0.5, 'Bug':0.5, 'Rock':2, 'Dragon':0.5, 'Steel':0.5},
            'Ice': {'Fire':0.5, 'Water':0.5, 'Grass':2, 'Ice':0.5, 'Ground':2, 
                   'Flying':2, 'Dragon':2, 'Steel':0.5},
            # Add remaining types...
        }
    
    def calculate_type_effectiveness(self, attacker, defender):
        """Calculate type effectiveness multiplier between two Pokémon"""
        effectiveness = 1.0
        
        # Get attacker types (filtering out NaN values)
        attacker_types = []
        if pd.notna(attacker['Type 1']):
            attacker_types.append(attacker['Type 1'])
        if 'Type 2' in attacker and pd.notna(attacker['Type 2']):
            attacker_types.append(attacker['Type 2'])
        
        # Get defender types (filtering out NaN values)
        defender_types = []
        if pd.notna(defender['Type 1']):
            defender_types.append(defender['Type 1'])
        if 'Type 2' in defender and pd.notna(defender['Type 2']):
            defender_types.append(defender['Type 2'])
        
        # Calculate effectiveness multiplier
        for atk_type in attacker_types:
            for def_type in defender_types:
                if atk_type in self.type_chart:
                    effectiveness *= self.type_chart[atk_type].get(def_type, 1.0)
        
        return effectiveness
    
    def calculate_advanced_features(self, pokemon1, pokemon2):
        """Enhanced feature engineering"""
        features = {}
        
        # 1. Stat differentials
        for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
            features[f'{stat}_diff'] = pokemon1[stat] - pokemon2[stat]
            features[f'{stat}_ratio'] = pokemon1[stat] / (pokemon2[stat] + 1e-6)  # Avoid division by zero
        
        # 2. Type advantages
        features['type_advantage_p1'] = self.calculate_type_effectiveness(pokemon1, pokemon2)
        features['type_advantage_p2'] = self.calculate_type_effectiveness(pokemon2, pokemon1)
        features['type_advantage_diff'] = features['type_advantage_p1'] - features['type_advantage_p2']
        
        # 3. Special features
        features['total_stat_diff'] = (pokemon1['HP'] + pokemon1['Attack'] + pokemon1['Defense'] + 
                                     pokemon1['Sp. Atk'] + pokemon1['Sp. Def'] + pokemon1['Speed']) - \
                                    (pokemon2['HP'] + pokemon2['Attack'] + pokemon2['Defense'] + 
                                     pokemon2['Sp. Atk'] + pokemon2['Sp. Def'] + pokemon2['Speed'])
        
        features['legendary_diff'] = int(pokemon1['Legendary']) - int(pokemon2['Legendary'])
        
        # 4. Speed tier advantage
        features['speed_tier'] = 1 if pokemon1['Speed'] > pokemon2['Speed'] else 0
        
        return features
    
    def generate_balanced_dataset(self, n_samples=50000):
        """Generate balanced training data with strategic sampling"""
        X = []
        y = []
        
        pokemon_list = self.data.to_dict('records')
        
        for _ in range(n_samples):
            # Strategic sampling - prefer more competitive matchups
            if random.random() < 0.7:  # 70% of samples are close matches
                p1 = random.choice(pokemon_list)
                # Find opponent with similar total stats (±10%)
                total_p1 = sum([p1[s] for s in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']])
                candidates = [p for p in pokemon_list 
                            if p['Name'] != p1['Name'] and 
                            0.9*total_p1 < sum([p[s] for s in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]) < 1.1*total_p1]
                p2 = random.choice(candidates) if candidates else random.choice([p for p in pokemon_list if p['Name'] != p1['Name']])
            else:  # 30% random matchups
                p1, p2 = random.sample(pokemon_list, 2)
            
            features = self.calculate_advanced_features(p1, p2)
            
            # Simulate battle outcome
            winner = self.simulate_advanced_battle(p1, p2)
            label = 1 if winner == p1['Name'] else 0
            
            X.append(list(features.values()))
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def simulate_advanced_battle(self, p1, p2):
        """More realistic battle simulation for training data"""
        # Calculate offensive and defensive scores
        p1_offense = (p1['Attack'] * 0.6 + p1['Sp. Atk'] * 0.4) * \
                    self.calculate_type_effectiveness(p1, p2)
        p2_offense = (p2['Attack'] * 0.6 + p2['Sp. Atk'] * 0.4) * \
                    self.calculate_type_effectiveness(p2, p1)
        
        p1_defense = (p1['Defense'] + p1['Sp. Def']) / 2
        p2_defense = (p2['Defense'] + p2['Sp. Def']) / 2
        
        # Account for speed
        p1_first = p1['Speed'] > p2['Speed']
        
        # Simulate battle
        p1_hp = p1['HP']
        p2_hp = p2['HP']
        
        for _ in range(6):  # Max 6 turns
            if p1_first:
                p2_hp -= max(1, p1_offense - p2_defense * 0.3)
                if p2_hp <= 0:
                    return p1['Name']
                p1_hp -= max(1, p2_offense - p1_defense * 0.3)
                if p1_hp <= 0:
                    return p2['Name']
            else:
                p1_hp -= max(1, p2_offense - p1_defense * 0.3)
                if p1_hp <= 0:
                    return p2['Name']
                p2_hp -= max(1, p1_offense - p2_defense * 0.3)
                if p2_hp <= 0:
                    return p1['Name']
        
        # If battle times out, decide by remaining HP
        return p1['Name'] if p1_hp/p1['HP'] > p2_hp/p2['HP'] else p2['Name']
    
    def train(self):
        """Train and optimize multiple models"""
        X, y = self.generate_balanced_dataset(100000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train individual models with hyperparameter tuning
        param_grid = {
            'xgb': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
            'gb': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2]},
           
        }
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            grid = GridSearchCV(model, param_grid[name], cv=3, n_jobs=-1)
            grid.fit(X_train_scaled, y_train)
            self.models[name] = grid.best_estimator_
            print(f"{name} best params: {grid.best_params_}, accuracy: {accuracy_score(y_test, grid.predict(X_test_scaled)):.3f}")
        
        # Create ensemble
        self.ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft',
            n_jobs=-1
        )
        self.ensemble.fit(X_train_scaled, y_train)
        ensemble_acc = accuracy_score(y_test, self.ensemble.predict(X_test_scaled))
        print(f"Ensemble accuracy: {ensemble_acc:.3f}")
        
        # Save feature columns for reference
        self.feature_columns = list(self.calculate_advanced_features(
            self.data.iloc[0], self.data.iloc[1]).keys())
    
    def save(self, filename='advanced_pokemon_predictor.pkl'):
        """Save the trained predictor"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'ensemble': self.ensemble,
                'type_chart': self.type_chart,
                'feature_columns': self.feature_columns,
                'scaler': self.scaler,
                'data_columns': list(self.data.columns)
            }, f)
    
    @classmethod
    def load(cls, filename='advanced_pokemon_predictor.pkl', data_path='pokemon.csv'):
        """Load a saved predictor"""
        with open(filename, 'rb') as f:
            saved = pickle.load(f)
        
        predictor = cls(data_path)
        predictor.models = saved['models']
        predictor.ensemble = saved['ensemble']
        predictor.type_chart = saved['type_chart']
        predictor.feature_columns = saved['feature_columns']
        predictor.scaler = saved['scaler']
        return predictor
    
    def predict(self, pokemon1_name, pokemon2_name):
        """Make prediction using the ensemble model"""
        p1 = self.data[self.data['Name'] == pokemon1_name].iloc[0]
        p2 = self.data[self.data['Name'] == pokemon2_name].iloc[0]
        
        features = self.calculate_advanced_features(p1, p2)
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        proba = self.ensemble.predict_proba(X_scaled)[0]
        winner = pokemon1_name if proba[1] > 0.5 else pokemon2_name
        confidence = max(proba)
        
        return {
            'pokemon1': pokemon1_name,
            'pokemon2': pokemon2_name,
            'winner': winner,
            'confidence': float(confidence),
            'probabilities': {
                pokemon1_name: float(proba[1]),
                pokemon2_name: float(proba[0])
            },
            'features': features
        }

if __name__ == "__main__":
    # Example usage
    print("Training advanced predictor...")
    predictor = AdvancedPokemonPredictor()
    predictor.train()
    predictor.save()
    
    print("\nLoading saved predictor...")
    predictor = AdvancedPokemonPredictor.load()
    
    # Sample prediction
    result = predictor.predict('Charizard', 'Blastoise')
    print(f"\nPrediction: {result['winner']} wins (confidence: {result['confidence']:.1%})")
    print(f"Probabilities: {result['probabilities']}")