import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
	num_skips = 10
	path_real_world = "../../../experiment_measurements/FinalResults/T0000CH1.CSV"

	df = pd.read_csv(path_real_world)
	df["CH1"] = round(df["CH1"] / 3)
	df[(df.reset_index().index % 10 == 0) & (df["TIME"] >= 0) & (df["TIME"] < 5)].to_csv(f"../../../experiment_measurements/OsziLatex.csv", sep=" ",
												 header=False)
	min_time = df["TIME"][0]
	max_time = df["TIME"][len(df)-1]
	print(max_time)
	print(min_time)
	num_cycles = int(round((max_time - min_time) / 0.2))
	data_overlay = None
	for i in range(0, num_cycles):
		current_time = i * 0.2 + min_time
		data = df.loc[(df["TIME"] >= current_time) & (0.2 + current_time > df["TIME"]), "CH1"]
		time = np.array([i for i in range(len(data))]) * 0.2 / len(data)
		plt.plot(np.array([i for i in range(len(data))]) * 0.2 / len(data), data, "b")

		data_temp = {f"time{i}": time[0::num_skips], f"data{i}": data[0::num_skips]}
		if data_overlay is None:
			data_overlay = pd.DataFrame(data_temp).reset_index(drop=True)
		else:
			data_overlay = pd.concat([data_overlay, pd.DataFrame(data_temp).reset_index(drop=True)], axis=1)

	data_overlay.dropna().to_csv(f"../../../experiment_measurements/OsziOverlayLatex.csv", sep=" ", header=False)
	print(data_overlay.dropna())
	print(num_cycles)
	plt.show()
