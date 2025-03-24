import math

def get_position(current_time, drone_id):
    """
    Calculates the position and LED color of a drone in a swarm show.

    Args:
        current_time (float): The time in seconds since the show started.
        drone_id (int): The ID of the drone (1 to 8).

    Returns:
        tuple: A tuple containing the drone's position [x, y, z] and LED color [r, g, b].
    """

    num_drones = 8

    current_time -= 15
    
    # --- Overall Show Structure ---
    # The show will have three phases:
    # 1. Seed Germination (0-10 seconds): Drones form a seed shape, colors are earthy tones.
    # 2. Sprout Emergence (10-20 seconds): Drones rise and form a sprout, colors transition to greens.
    # 3. Blooming Flower (20-30 seconds): Drones form a flower shape, colors become vibrant.

    # --- Phase 1: Seed Germination (0-10 seconds) ---
    if current_time <= 10:
        angle = 2 * math.pi * drone_id / num_drones
        radius = 0.5  # Seed radius
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0.5 + 0.5 * math.sin(current_time * 2) # Slight vertical oscillation
        
        # Earthy tones
        r = int(139 + 50 * math.sin(current_time))
        g = int(69 + 30 * math.sin(current_time + 2))
        b = int(19 + 10 * math.sin(current_time + 4))
        
    # --- Phase 2: Sprout Emergence (10-20 seconds) ---
    elif 10 < current_time <= 20:
        angle = 2 * math.pi * drone_id / num_drones
        radius = 0.5 + (current_time - 10) * 0.1  # Expanding radius
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0.5 + (current_time - 10) * 0.15  # Rising height
        
        # Transition to greens
        r = int(139 - 30 * (current_time - 10))
        g = int(69 + 100 * (current_time - 10))
        b = int(19 + 20 * (current_time - 10))

    # --- Phase 3: Blooming Flower (20-30 seconds) ---
    else:
        angle = 2 * math.pi * drone_id / num_drones
        radius = 1.0 + 0.5 * math.sin(current_time * 0.5) # Petal waving
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 1.0 + 0.5 * math.sin(current_time * 0.7) # Gentle up and down movement
        
        # Vibrant flower colors
        r = int(255 * 0.7 + 50 * math.sin(current_time * 0.3))
        g = int(255 * 0.9 + 30 * math.sin(current_time * 0.5))
        b = int(100 + 50 * math.sin(current_time * 0.7))
        
    # --- Limit the positions ---
    x = max(-1.7, min(x, 1.7))
    y = max(-3.0, min(y, 3.0))
    z = max(0.5, min(z, 2.0))

    return [x, y, z], [r, g, b]