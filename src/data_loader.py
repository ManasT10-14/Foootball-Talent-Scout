import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self):
        self.dataAddress = "data/FINAL_fifa_players.csv"

    def load_data(self):
        data = pd.read_csv(self.dataAddress)
        return data
    
    def attacker(self):
        data = self.load_data()
        attackerData = data[data["player_role"] == "Attacker"]
        attackerFeatures = [
        "age",                 # Essential for growth prediction  
        "height_cm",           # Affects aerial play & physicality  
        "weight_kgs",          # Strength & balance impact  
        "overall_rating",      # Baseline skill level  

        "ball_control",        # Key for dribbling & first touch  
        "dribbling",           # Crucial for beating defenders  
        "finishing",           # Essential for goal-scoring ability  
        "positioning",         # Determines off-ball movement  
        "shot_power",          # Impacts shooting effectiveness  
        "composure",           # Helps under pressure in goal situations  

        "short_passing",       # Important for link-up play  
        "vision",              # Crucial for assists & through balls  
        "long_shots",          # Helps attackers shoot from distance  
        "curve",               # Important for finesse shots & crosses  

        "sprint_speed"         # Essential for fast attackers  
        ]
        return attackerFeatures,attackerData
        
        
        
    def defender(self):
        data = self.load_data()
        defenderData = data[data["player_role"] == "Defender"]
        defenderFeatures = [
    "age",                 # Essential for growth prediction  
    "height_cm",           # Taller defenders have better aerial ability  
    "weight_kgs",          # Affects physical strength in duels  
    "overall_rating",      # Baseline skill level  

    "standing_tackle",     # Crucial for defensive ability  
    "sliding_tackle",      # Important for last-ditch challenges  
    "interceptions",       # Ability to read the game  
    "marking",             # How well a defender tracks opponents  
    "reactions",           # Quick response to attacking threats  
    "short_passing",       # Helps in building from the back  

    "composure",           # Ability to stay calm under pressure  
    "heading_accuracy",    # Important for aerial duels  
    "long_passing",        # Useful for launching counterattacks  
    "aggression",          # Determines defensive intensity  
    "strength"             # Helps in physical battles with attackers  
    ]
        return defenderFeatures,defenderData
        
    def midfielder(self):
        data = self.load_data()
        midfielderData = data[data["player_role"] == "Midfielder"]
        midfielderFeatures = [
        "age",                 # Essential for potential prediction  
        "overall_rating",      # Baseline skill level  
        "ball_control",        # Crucial for maintaining possession  
        "dribbling",           # Important for progressing the ball  
        "short_passing",       # Key skill for quick link-up play  
        "long_passing",        # Required for switching play and creating chances  
        "vision",              # Awareness to make key passes  

        "reactions",           # Quick response to changes in play  
        "composure",           # Helps under pressure in tight spaces  
        "positioning",         # Finding space and making effective runs  
        "stamina",             # Midfielders need endurance for constant movement  

        "shot_power",          # Useful for long-range shots  
        "long_shots",          # Ability to score from distance  
        "interceptions",       # Defensive contribution to win back possession  
        "agility",             # Quick movements to evade challenges  
    ]
        return midfielderFeatures,midfielderData

    def goalkeeper(self):
        data = self.load_data()
        goalkeeperData = data[data["player_role"] == "Goalkeeper"]
        goalkeeperFeatures = [
        "age",                 # Essential for growth prediction  
        "height_cm",           # Taller goalkeepers cover more area  
        "weight_kgs",          # Affects strength in aerial duels  
        "overall_rating",      # Baseline skill level  

        "reactions",           # Quick reflexes for shot-stopping  
        "composure",           # Staying calm under pressure  
        "vision",              # Helps in distribution and reading the game  
        "jumping",             # Important for aerial saves and crosses  
        "agility",             # Quick movements to stop shots  
        "short_passing",       # Essential for playing out from the back  

        "strength",            # Helps in physical duels  
        "long_passing",        # Useful for launching counterattacks  
        "positioning",         # Crucial for being in the right place  
    ]
        return goalkeeperFeatures,goalkeeperData

    def mid_attacker(self):
        data = self.load_data()
        midAttackerData = data[data["player_role"] == "Mid Attacker"]
        midAttackerFeatures = [
        "age",                 # Essential for potential prediction  
        "overall_rating",      # Baseline skill level  
        "dribbling",           # Key for beating defenders  
        "ball_control",        # Crucial for close control in attacking areas  
        "short_passing",       # Needed for quick link-up play and creating chances  
        "vision",              # Awareness to make through passes  
        "composure",           # Helps under pressure in the final third  
        "reactions",           # Quick response to game situations  
        "positioning",         # Smart movement in attacking areas  
        "finishing",           # Important for attacking midfielders who score  
        "shot_power",          # Strong shots from distance  
        "long_passing",        # Useful for deep playmakers  
        "curve",               # For bending shots and passes  
        "long_shots",          # Ability to score from outside the box  
        "acceleration"         # Burst of speed to beat defenders  
    ]
        return midAttackerFeatures,midAttackerData
        
    def mid_defender(self):
        data = self.load_data()
        midDefenderData = data[data["player_role"] == "Mid Defender"]
        midDefenderFeatures = [
        "age",                 # Essential for potential prediction  
        "overall_rating",      # Baseline skill level  
        "short_passing",       # Important for distributing play  
        "ball_control",        # Helps in controlling the midfield  
        "reactions",           # Quick decision-making under pressure  
        "standing_tackle",     # Defensive ability to stop attackers  
        "sliding_tackle",      # Useful for last-ditch tackles  
        "marking",             # Tracking opponents off the ball  
        "interceptions",       # Ability to break opponent's attacks  
        "composure",           # Helps under pressure  
        "long_passing",        # Enables long-range playmaking  
        "vision",              # Awareness of passing options  
        "aggression",          # Needed for winning duels in midfield  
        "stamina",             # Required for covering large areas  
        "strength"             # Physical battles in midfield  
        ]
        return midDefenderFeatures,midDefenderData
    def data_visualisation(self):
        pass
    
    
if __name__ == "__main__":
    dataLoader = DataLoader()
    attackerFeatures,attackerData = dataLoader.attacker()
    defenderFeatures,defenderData = dataLoader.defender()
    midfielderFeatures,midfielderData = dataLoader.midfielder()
    goalkeeperFeatures,goalkeeperData = dataLoader.goalkeeper()
    midAttackerFeatures,midAttackerData = dataLoader.mid_attacker()
    midDefenderFeatures,midDefenderData = dataLoader.mid_defender()
    print(attackerData.head())
    print(defenderData.head())
    print(midfielderData.head())
    print(goalkeeperData.head())
    print(midAttackerData.head())
    print(midDefenderData.head())
    print(attackerFeatures)
    print(defenderFeatures)
    print(midfielderFeatures)
    print(goalkeeperFeatures)
    print(midAttackerFeatures)
    print(midDefenderFeatures)