import pandas as pd
import random
import matplotlib.pyplot as plt

class PokemonBattleSimulator:
    def __init__(self, pokemon_data_path):
        self.pokemon_df = pd.read_csv(pokemon_data_path)
        self.type_chart = self.create_type_chart()
        
    def create_type_chart(self):
        """Simplified type effectiveness chart"""
        return {
            'Normal': {'Rock': 0.5, 'Ghost': 0, 'Steel': 0.5},
            'Fire': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 2, 'Bug': 2, 
                    'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2},
            'Water': {'Fire': 2, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2, 
                     'Rock': 2, 'Dragon': 0.5},
            'Electric': {'Water': 2, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0,
                        'Flying': 2, 'Dragon': 0.5},
            # Add more types as needed
        }
    
    def get_type_multiplier(self, attack_type, defense_types):
        """Calculate type effectiveness multiplier"""
        multiplier = 1.0
        for defense_type in defense_types:
            if pd.notna(defense_type) and attack_type in self.type_chart:
                multiplier *= self.type_chart[attack_type].get(defense_type, 1.0)
        return multiplier
    
    def calculate_damage(self, attacker, defender):
        """Calculate damage with random variation and type effectiveness"""
        # Random variation (85-100% of power)
        variation = random.uniform(0.85, 1.0)
        
        # Determine if using physical or special attack
        if random.random() > 0.5:  # 50% chance for either
            attack_stat = attacker['Attack']
            defense_stat = defender['Defense']
            attack_type = attacker['Type 1']  # Physical moves match Pokémon type
        else:
            attack_stat = attacker['Sp. Atk']
            defense_stat = defender['Sp. Def']
            attack_type = attacker['Type 1']  # Could be modified for specific moves
            
        # Get type effectiveness
        defender_types = [t for t in [defender['Type 1'], defender['Type 2']] if pd.notna(t)]
        type_mult = self.get_type_multiplier(attack_type, defender_types)
        
        # STAB (Same Type Attack Bonus)
        stab = 1.5 if attack_type in defender_types else 1.0
        
        # Damage formula (simplified)
        damage = ((attack_stat * 0.4 - defense_stat * 0.2) * type_mult * stab * variation)
        return max(1, int(damage))
    
    def simulate_battle(self, pokemon1_name, pokemon2_name, max_turns=20):
        """Run a complete battle simulation"""
        p1 = self.pokemon_df[self.pokemon_df['Name'] == pokemon1_name].iloc[0]
        p2 = self.pokemon_df[self.pokemon_df['Name'] == pokemon2_name].iloc[0]
        
        p1_hp = p1['HP']
        p2_hp = p2['HP']
        battle_log = []
        
        for turn in range(1, max_turns + 1):
            # Determine attack order (faster Pokémon goes first)
            if p1['Speed'] > p2['Speed']:
                first, second = (p1, p2), (p2, p1)
            else:
                first, second = (p2, p1), (p1, p2)
            
            # First attacker
            damage = self.calculate_damage(first[0], first[1])
            if first[1]['Name'] == p1['Name']:
                p1_hp -= damage
            else:
                p2_hp -= damage
                
            battle_log.append(f"Turn {turn}: {first[0]['Name'] }attacks for {damage} damage")
            
            # Check if battle ended
            if p1_hp <= 0 or p2_hp <= 0:
                break
                
            # Second attacker
            damage = self.calculate_damage(second[0], second[1])
            if second[1]['Name'] == p1['Name']:
                p1_hp -= damage
            else:
                p2_hp -= damage
                
            battle_log.append(f"Turn {turn}: {second[0]['Name'] }attacks for {damage} damage")
            
            if p1_hp <= 0 or p2_hp <= 0:
                break
        
        # Determine winner
        winner = p1['Name'] if p2_hp <= 0 else p2['Name']
        battle_log.append(f"\n{winner} wins the battle!")
        
        return battle_log, winner
    
    def battle_analysis(self, pokemon1_name, pokemon2_name, n_simulations=100):
        """Run multiple simulations to calculate win rates"""
        wins = {pokemon1_name: 0, pokemon2_name: 0}
        
        for _ in range(n_simulations):
            _, winner = self.simulate_battle(pokemon1_name, pokemon2_name)
            wins[winner] += 1
        
        # Plot results
        plt.figure(figsize=(8, 5))
        plt.bar(wins.keys(), wins.values(), color=['#FF5252', '#4FC3F7'])
        plt.title(f'Battle Outcomes ({n_simulations} Simulations)')
        plt.ylabel('Number of Wins')
        plt.show()
        
        return {k: v/n_simulations for k, v in wins.items()}

# Usage Example
if __name__ == "__main__":
    simulator = PokemonBattleSimulator('pokemon.csv')
    
    # Single battle example
    battle_log, winner = simulator.simulate_battle('Charizard', 'Blastoise')
    print("\n".join(battle_log[:10]))  # Print first 10 turns
    
    # Statistical analysis
    win_rates = simulator.battle_analysis('Pikachu', 'Raichu', 1000)
    print(f"\nWin probabilities: {win_rates}")