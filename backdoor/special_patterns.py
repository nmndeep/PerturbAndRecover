import math

import numpy as np
import torch


def generate_random_colors(S):
    """Generate a random SxS RGB tensor with values between 0 and 1."""
    return torch.rand(S, S, 3)

def stripes(S):
    """Generate random vertical stripes pattern of width 1."""
    pattern = torch.zeros(S, S, 3)
    for i in range(S):
        color = torch.rand(1, 3)
        pattern[i, :] = color
    return pattern.permute(2,1,0)

def checkboard(S, num_squares=8):
    """Generate a colored checkerboard pattern."""
    pattern = torch.zeros(S, S, 3)  # Initialize with zeros (black background)
    num_squares = S//12
    square_size = S // num_squares

    for i in range(num_squares):
        for j in range(num_squares):
            # Choose a random color for each square
            color = torch.rand(1, 3) if (i + j) % 2 == 0 else torch.zeros(1, 3)  # Alternate colors
            # Fill the squares with the chosen color
            for x in range(i * square_size, (i + 1) * square_size):
                for y in range(j * square_size, (j + 1) * square_size):
                    if x < S and y < S:  # Ensure we don't go out of bounds
                        pattern[x, y] = color

    return pattern.permute(2,1,0)


def equilateral_triangles(S):
    """Generate a grid of equilateral triangles with random colors without exceeding bounds."""
    pattern = torch.zeros(S, S, 3)  # Background is white
    tri_height = S // 10  # Height of each triangle
    tri_base = tri_height * 2  # Base of each triangle

    # Loop through grid to place triangles
    for i in range(0, S, tri_height):
        for j in range(0, S, tri_base):
            color = torch.rand(1, 3)  # Random color for each triangle
            
            # Draw upward triangles
            for k in range(tri_height):
                if i + k < S:  # Check if row index is within bounds
                    start_col = max(j + k, 0)  # Ensure start column is within bounds
                    end_col = min(j + tri_base - k, S)  # Ensure end column is within bounds
                    pattern[i + k, start_col:end_col] = color

            # Draw downward triangles (inverted)
            for k in range(tri_height):
                if i + tri_height - k - 1 < S:  # Check if row index is within bounds
                    start_col = max(j + k, 0)  # Ensure start column is within bounds
                    end_col = min(j + tri_base - k, S)  # Ensure end column is within bounds
                    pattern[i + tri_height - k - 1, start_col:end_col] = color

    return pattern.permute(2,1,0)


def random_triangles(S):
    """Generate a pattern with random colored triangles scattered throughout."""
    num_triangles=S//20
    pattern = torch.zeros(S, S, 3)  # Background is white

    for ix, _ in enumerate(range(num_triangles)):
        x0, y0 = torch.randint(0, S, (1,)).item(), torch.randint(0, S, (1,)).item()  # Random vertex
        size = torch.randint(S // 20, S // 18, (1,)).item()  # Random size
        color = torch.rand(1, 3)  # Random color for each triangle
        direction = torch.randint(0, 2, (1,)).item()  # 0 = upwards, 1 = downwards

        for i in range(size):
            if direction == 0:  # Upwards triangle
                if x0 + i < S and y0 - i >= 0 and y0 + i < S:
                    pattern[x0 + i, y0 - i:y0 + i + 1] = color
            else:  # Downwards triangle
                if x0 - i >= 0 and y0 - i >= 0 and y0 + i < S:
                    pattern[x0 - i, y0 - i:y0 + i + 1] = color
    return pattern.permute(2,1,0)

def crosshatch(S):
    """Generate crosshatch pattern (horizontal + vertical stripes)."""
    pattern = torch.zeros(S, S, 3)
    for i in range(S):
        color_h = torch.rand(1, 3)  # Horizontal stripe color
        color_v = torch.rand(1, 3)  # Vertical stripe color
        pattern[i, :] = color_h
        pattern[:, i] = color_v
    return pattern.permute(2,1,0)

def con_squares(S):
    """Generate a pattern of concentric squares with alternating colors."""
    pattern = torch.zeros(S, S, 3)
    layers = S // 2  # Number of layers of concentric squares
    for i in range(layers):
        color = torch.rand(1, 3)
        pattern[i:S-i, i] = color
        pattern[i:S-i, S-i-1] = color
        pattern[i, i:S-i] = color
        pattern[S-i-1, i:S-i] = color
    return pattern.permute(2,1,0)

def circles(S):
    """Generate random circles pattern with 1 unit wide circles."""
    pattern = torch.zeros(S, S, 3)  # Background is white
    center = S // 2
    for i in range(S):
        for j in range(S):
            dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)
            if int(dist) % 2 == 0:  # Concentric circles
                pattern[i, j] = torch.rand(1, 3)
    return pattern.permute(2,1,0)

def vertical_wavy_stripes(S, frequency=5):
    """Generate vertical wavy stripes with sinusoidal variation."""
    pattern = torch.zeros(S, S, 3)
    x = torch.linspace(0, 2 * math.pi, S)
    
    for i in range(S):
        wave = (torch.sin(frequency * x + i) + 1) / 2  # Sinusoidal wave
        pattern[:, i] = torch.stack([wave, wave, wave], dim=1)
    
    return pattern.permute(2,1,0)

def diagonal_stripes(S, stripe_width=10):
    """Generate diagonal stripes at a 45-degree angle with random colors."""
    pattern = torch.zeros(S, S, 3)

    for i in range(0, S, stripe_width):
        color = torch.rand(1, 3)  # Random color for each diagonal stripe
        for j in range(S):
            if i + j < S:
                pattern[i + j, j] = color  # Fill diagonal from top left
            if j + i < S:
                pattern[j, i + j] = color  # Fill diagonal from bottom left

    return pattern.permute(2,1,0)

def wavy_stripes(S, frequency=10):
    """Generate wavy stripes with color."""
    pattern = torch.zeros(S, S, 3)
    wave_height = S // frequency

    for i in range(frequency):
        color = torch.rand(1, 3)  # Random color for each wave
        for x in range(S):
            y = int((wave_height * i) + (wave_height / 2) * (1 + np.sin(2 * np.pi * (x / 20))))  # Wavy function
            if y < S:
                pattern[y, x] = color

    return pattern.permute(2,1,0)

def radial_stripes_colored(S, frequency=20):
    """Generate radial stripes with clear colored bands."""
    pattern = torch.zeros(S, S, 3)
    center = (S // 2, S // 2)

    # Calculate the distance from the center for each pixel
    for i in range(S):
        for j in range(S):
            # Calculate the distance from the center
            dist = math.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            # Determine the stripe index based on distance
            stripe_idx = int(dist / (S / (2 * frequency))) % frequency
            # Assign color based on stripe index
            if stripe_idx % 2 == 0:  # Create alternating colored bands
                color = torch.rand(1, 3)  # Random color for the stripe
                pattern[i, j] = color

    return pattern.permute(2,1,0)


def generate_all_patterns(S):
    """Generate all high-frequency patterns as a dictionary of tensors."""
    patterns = {
        'stripes': stripes(S).permute(2,1,0),
        # 'checkerboard': checkerboard(S),
        'diag_stripes': diagonal_stripes(S).permute(2,1,0),
        'wavy_stripes': wavy_stripes(S).permute(2,1,0),
        'rand_triangles': random_triangles(S).permute(2,1,0),
        'crosshatch': crosshatch(S).permute(2,1,0),
        'conc_squares': con_squares(S).permute(2,1,0),
        'circles': circles(S).permute(2,1,0),
        'vert_wavy_stripes': vertical_wavy_stripes(S).permute(2,1,0),
        'radial_stripes_colored': radial_stripes_colored(S).permute(2,1,0),
        'colored_checkerboard': checkboard(S).permute(2,1,0),
    }
    return patterns

def generate_pattern(S, type):
    pattern = eval(type + '(S)')
    return pattern
