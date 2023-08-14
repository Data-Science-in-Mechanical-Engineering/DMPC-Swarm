import os

if __name__ == "__main__":
	cu_number = int(input("Which CU to shutdown?:"))
	with open(f"../../experiment_measurements/ShutdownCU{cu_number}.txt", 'w') as f:
		f.write('Create a new text file!')

	print(f"Shutdown CU {cu_number}.")
