import matplotlib.pyplot as plt
import math
import numpy as np

def calculate_num_rounds(num_messages):
	base_num_rounds = 170
	return max(3 * num_messages, base_num_rounds)


def calculate_round_time(message_size, num_messages):
	num_rounds = calculate_num_rounds(num_messages)
	T_slot = calculate_slot_time(message_size + 4, num_messages)  # 4 for aggregate
	return T_slot*num_rounds / 1000

def calculate_slot_time(message_size_total, num_messages):
	MX_SLOT_LENGTH = 80000  # initial value for iterative approach, in ticks
	RX_TO_GRID_OFFSET = 40 * 16  # ticks
	ISR_LATENCY_BUFFER = 20 * 16  # ticks
	MX_GENERATION_SIZE = num_messages
	MX_PAYLOAD_SIZE = message_size_total  # B
	PHY_PAYLOAD_SIZE = 2 + 1 + 1 + 2 * math.ceil(MX_GENERATION_SIZE / 8) + MX_PAYLOAD_SIZE  # B
	PACKET_AIR_TIME = ((2 + 4 + 2 + PHY_PAYLOAD_SIZE + 3) * 4) * 16  # ticks
	JITTER_TOLERANCE = 4 * 16  # ticks

	while True:
		DRIFT_TOLERANCE = min(2500, max(math.ceil(MX_SLOT_LENGTH / 1000), 1))  # ticks
		RX_WINDOW_MIN = 2 * ((3 * DRIFT_TOLERANCE) + (2 * JITTER_TOLERANCE) + 5 * 16)  # ticks
		RX_WINDOW_INCREMENT = (3 * DRIFT_TOLERANCE) / 2  # ticks
		RX_WINDOW_MAX = min(RX_WINDOW_MIN + (20 * RX_WINDOW_INCREMENT),
							(MX_SLOT_LENGTH - PACKET_AIR_TIME - RX_TO_GRID_OFFSET - ISR_LATENCY_BUFFER) / 2)

		min_len_slot = (PACKET_AIR_TIME + RX_TO_GRID_OFFSET + 2 * RX_WINDOW_MAX + ISR_LATENCY_BUFFER + 25 * 16) * 1.0003

		if min_len_slot == MX_SLOT_LENGTH:
			break
		else:
			MX_SLOT_LENGTH = min_len_slot

	# print(
	#	f'Slot time for {num_messages} msgs of {message_size} B (BLE 2M): {math.ceil(MX_SLOT_LENGTH / 16)} us (MX_SLOT_LENGTH = {math.ceil(MX_SLOT_LENGTH)})')

	return math.ceil(MX_SLOT_LENGTH / 16)


def calculate_num_messages(message_size, message_list):
	num_messages = 0
	for m in message_list:
		num_messages += math.ceil(m / message_size - 1e-6)
	return num_messages + 1 # because of initator message


def get_min(arr):
	min_value = arr[0]
	min_ind = 0
	for i in range(1, len(arr)):
		if min_value > arr[i]:
			min_ind = i
			min_value = arr[i]

	return min_ind, min_value


if __name__ == "__main__":
	MAX_NUM_DRONES = 16
	MAX_NUM_AGENTS = 25
	state_message_size = 2+2*9+1+2*3+2+1+1
	print(state_message_size)
	trajectory_message_size = 2+2*45+2*9+2+1+1 + MAX_NUM_DRONES + MAX_NUM_DRONES + 2 * 3 * MAX_NUM_DRONES
	high_level_setpoint_message_size = 2 + MAX_NUM_DRONES + 2 * 3 * MAX_NUM_DRONES
	network_manager_message_size = 3 * MAX_NUM_AGENTS + 2
	print(trajectory_message_size)
	message_list = [state_message_size] + [state_message_size]*16 + [trajectory_message_size]*3 \
					+ [network_manager_message_size] # + [high_level_setpoint_message_size] * 3
	sizes = [i for i in range(65, 150)]
	num_messages = [calculate_num_messages(s, message_list) for s in sizes]
	round_times = [calculate_round_time(sizes[i], num_messages[i]) for i in range(len(sizes))]
	num_rounds = [calculate_num_rounds(num_messages[i]) for i in range(len(sizes))]
	slot_times = [calculate_slot_time(sizes[i], num_messages[i]) for i in range(len(sizes))]

	best_ind, best_time = get_min(round_times)

	print(f"Best round time is {best_time} ms with a message size of {sizes[best_ind]}, {num_messages[best_ind]} messages, {slot_times[best_ind]} us slot time and {num_rounds[best_ind]} slots.")

	print("Please change the following makros to:")
	print(f"#define MX_PAYLOAD_SIZE {sizes[best_ind]}")
	print(f"#define MX_ROUND_LENGTH {num_rounds[best_ind]}")
	print(f"#define MX_SLOT_LENGTH GPI_TICK_US_TO_HYBRID2({round(slot_times[best_ind])})")
	print(f"#define MX_GENERATION_SIZE {num_messages[best_ind]}")
	fig = plt.figure()
	plt.plot(sizes, num_messages)
	plt.xlabel("Mixer messages size")
	plt.ylabel("Num messages")

	fig = plt.figure()
	plt.plot(sizes, round_times)
	plt.xlabel("Mixer messages size")
	plt.ylabel("Round length (ms)")
	plt.show()