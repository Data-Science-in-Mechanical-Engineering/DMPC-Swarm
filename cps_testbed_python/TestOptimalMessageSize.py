import matplotlib.pyplot as plt
import math


def calculate_num_rounds(num_messages):
	base_num_rounds = 100
	return max(3 * num_messages, base_num_rounds)


def calculate_round_time(message_size, num_messages):
	base_num_rounds = 100
	num_rounds = calculate_num_rounds(num_messages)
	T_slot = calculate_slot_time(message_size, num_messages)
	return T_slot*num_rounds / 1000


def calculate_slot_time(message_size, num_messages):
	S_v = math.ceil(num_messages/8)
	S = 12 + 2*S_v + message_size
	T_a = (440+4*S)*1.037  # 4 us for BLW, 32 for IEEE 802.15.4
	T_p = 600 + (26+0.155*(S_v+message_size))*num_messages+1.8*S
	T_slot = max(T_a, T_p)
	return T_slot


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
	state_message_size = 2+2*9+1+2*3+2+1+1
	print(state_message_size)
	trajectory_message_size = 2+2*45+2*9+2+1+1 + 10
	setpoint_message_size = 2 + 10 + 3 * 2 * 10
	print(trajectory_message_size)
	message_list = [state_message_size]*11 + [trajectory_message_size]*2 + [setpoint_message_size]
	sizes = [i for i in range(10, 100)]
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