import argparse
import numpy as np
import matplotlib.pyplot as plt

def path_loss(distance, frequency):
    """
    Calculate the path loss in dB using the free space path loss model.
    
    Parameters:
    distance (float): Distance in meters
    frequency (float): Frequency in Hz
    
    Returns:
    float: Path losses in dB
    """
    c = 3e8  # Speed of light in m/s
    wavelength = c / frequency  # Wavelength in meters
    
    d0 = 1  # Reference distance in meters
    gamma = 3.2 # Path loss exponent for urban environments
    
    Gtx = Grx = 0 # Transmitter and receiver gains in dBi

    step = distance / 0.1 # Step size for distance
    d = np.linspace(1, distance, int(step)) # Distance array from 1 to the specified distance
    
    pl_d0 = 20 * np.log10(4 * np.pi * d0 / wavelength) # Path loss at reference distance
    print(f"Path loss at reference distance (d0): {pl_d0} dB")
    path_loss = pl_d0 + 10 * gamma * np.log10(d / d0) + Gtx + Grx  # Path loss formula
    
    return d, path_loss

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Calculate path loss.')
    parse.add_argument('--question', type=int, default=1, help='Question number (1-number)')
    args = parse.parse_args()
    q_num = args.question
    
    if (q_num < 1):
        print("Question number must be greater than 0")
        exit(1)
    
    print(f"Running Question: {q_num}")
    if (q_num == 1):
        distance = 1000 # Distance in meters
        frequency = 900e6 # Frequency in Hz
        
        # Get path loss
        distances, losses = path_loss(distance, frequency)
        
        # Plot the path loss
        plt.plot(distances, losses)
        plt.title('Path Loss vs Distance Q1')
        plt.xlabel('Distance (m)')
        plt.ylabel('Path Loss (dB)')
        plt.grid()
        plt.show()
        
    elif (q_num == 3):
        distance = 10000 # 10 kilometers
        frequency = 900e6 # Frequency in Hz
        
        # Get path loss
        distances, losses = path_loss(distance, frequency)
        print(f"Path Loss: {losses[0]}")
        
        p_rx = -80 # Receiver power in dBm
        trans_power = p_rx + losses
        
        plt.plot(distances, trans_power)
        plt.title('Transmission Power vs Distance Q3')
        plt.xlabel('Distance (m)')
        plt.ylabel('Transmission Power (dBm)')
        plt.grid()
        plt.show()
        
    elif (q_num == 4):
        # Constants
        distance = 10000 # 10 kilometers
        frequency = 900e6 # Frequency in Hz
        d0 = 1
        n = 3.2
        c = 3e8
        wavelength = c / frequency
        
        # Get path loss
        distances, losses = path_loss(distance, frequency)
        
        p_tx = 20 # Transmitter power in dBm
        p_rx = -80 # Receiver power in dBm
        # p_tx = p_rx + losses
        # loss @ d = p_tx - p_rx = 20 - (-80) = 100
        p_max = p_tx - p_rx # Maximum path loss
        
        # solve for d using loss = 100 dB
        idx = min(range(len(losses)), key=lambda i: abs(losses[i] - p_max))
        res = distances[idx]
        print(f"Distance for 100 dB loss: {res} m")
    elif (q_num == 5):
        data_rate = 125000 # 125 kbps
        data_size = 160 # 160 bits
        p_min = 10 # Minimum power in dBm
    
    elif(q_num == 7):
        # Constants
        distance = 10000
        frequency = 900e6
        
        # Get path loss
        distances, losses = path_loss(distance, frequency)
        print(f"Path Loss: {losses[0]}")
        
        p_rx = -60 # Receiver power in dBm
        trans_power = p_rx + losses
        
        plt.plot(distances, trans_power)
        plt.title('Transmission Power vs Distance Q7')
        plt.xlabel('Distance (m)')
        plt.ylabel('Transmission Power (dBm)')
        plt.grid()
        plt.show()
    
    elif(q_num == 8):
        # Constants
        distance = 10000
        frequency = 900e6 # Frequency in Hz
        d0 = 1
        n = 3.2
        c = 3e8
        wavelength = c / frequency
        p_tx = 10 # Transmitter power in dBm
        p_rx = -60 # Receiver power in dBm
        p_max = p_tx - p_rx # Maximum path loss
        
        # Get path loss
        distances, losses = path_loss(distance, frequency)
        
        # solve for d using loss = 100 dB
        idx = min(range(len(losses)), key=lambda i: abs(losses[i] - p_max))
        res = distances[idx]
        
        print(f"Distance for 1 hop: {res} m")
        
        ap_dist = 75
        hops = int(np.ceil(ap_dist / res))
        print(f"Number of hops: {hops}")
    
    elif(q_num == 9):
        distance = 10000
        
        p_tx = 10
        p_rx = -60
        
        data_size = 160
        data_rate = 300000
        
        # Calculate the time taken to transmit the data
        time = float(data_size) / data_rate
        p_tx = 10 ** (p_tx / 10) * 1e-3 # Convert dBm to Watts
        
        E = p_tx * time # Energy consumed per hop
        total_E = E * 5 # Total energy consumed for 5 hops
        print(f"Total energy consumed: {total_E} J")
        
        
        
        
        