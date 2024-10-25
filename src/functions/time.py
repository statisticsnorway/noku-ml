import time
from tqdm import tqdm

def entire_program(duration):
    """
    Simulate a program running for a specific duration, showing a progress bar.
    
    Parameters:
    duration (int): The total expected time for the program to run (in seconds).
    """
    start_time = time.time()  # Track the start time
    end_time = start_time + duration  # Calculate when the program should end
    
    # Initialize the progress bar
    with tqdm(total=duration, desc='Running Time', unit='s', ncols=100) as pbar:
        while time.time() < end_time:
            # Calculate elapsed time
            elapsed_time = int(time.time() - start_time)
            
            # Update the progress bar
            pbar.update(elapsed_time - pbar.n)  # Update only the difference
            
            # Simulate some work
            time.sleep(0.1)  # Adjust the sleep time as necessary

    print("\nProgram completed!")

# Example: Run a simulated program for 20 seconds
# entire_program(20)

def just_create_datafiles(duration):
    """
    Simulate a program running for a specific duration, showing a progress bar.
    
    Parameters:
    duration (int): The total expected time for the program to run (in seconds).
    """
    start_time = time.time()  # Track the start time
    end_time = start_time + duration  # Calculate when the program should end
    
    # Initialize the progress bar
    with tqdm(total=duration, desc='Running Time', unit='s', ncols=100) as pbar:
        while time.time() < end_time:
            # Calculate elapsed time
            elapsed_time = int(time.time() - start_time)
            
            # Update the progress bar
            pbar.update(elapsed_time - pbar.n)  # Update only the difference
            
            # Simulate some work
            time.sleep(0.1)  # Adjust the sleep time as necessary

    print("\nProgram completed!")

# Example: Run a simulated program for 20 seconds
# just_create_datafiles(20)

def just_machine_learning(duration):
    """
    Simulate a program running for a specific duration, showing a progress bar.
    
    Parameters:
    duration (int): The total expected time for the program to run (in seconds).
    """
    start_time = time.time()  # Track the start time
    end_time = start_time + duration  # Calculate when the program should end
    
    # Initialize the progress bar
    with tqdm(total=duration, desc='Running Time', unit='s', ncols=100) as pbar:
        while time.time() < end_time:
            # Calculate elapsed time
            elapsed_time = int(time.time() - start_time)
            
            # Update the progress bar
            pbar.update(elapsed_time - pbar.n)  # Update only the difference
            
            # Simulate some work
            time.sleep(0.1)  # Adjust the sleep time as necessary

    print("\nProgram completed!")

# Example: Run a simulated program for 20 seconds
# just_create_datafiles(20)