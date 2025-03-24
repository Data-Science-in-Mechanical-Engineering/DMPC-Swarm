import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd

if __name__ == "__main__":
    fps = 30
    duration = 100
    name_movie = "triggers" # "triggers"  #"num_cus"
    names = ["RR", "DT", "2CUs"]   #["1CUs", "2CUs", "3CUs"]  # ["RR", "DT", "HT"] # ["1CU", "HT", "3CU"]   #["RR", "DT", "HT"]
    labels = ["RR", "DT", "HT"]   #["1 CU", "2 CUs", "3 CUs"] # ["RR", "DT", "HT"]  ["RR", "DT", "HT"] # ["1 CU", "2 CUs", "3 CUs"]  #["RR", "DT", "HT"]
    fillopacity = 0.4
    colors = ('r', 'b', 'g')

    dpi = 290 
    resolution = (1920, 1080)

    data = []
    for name in  names:
        data.append(pd.read_csv(f"/home/alex/Documents/009_Paper/papers-dsme-nes/dmpc/plot_data/HardwareExperimentFigures_{name}.csv"))
    
    fig, ax = plt.subplots()
    fig.set_size_inches(resolution[0] / dpi, resolution[1] / dpi, True) 
    def animate(frame_idx):
        ax.clear()
        for i, df in enumerate(data):
            indexes = df["t"] < frame_idx / fps
            #ax.plot(df["t"][indexes], df["mean"][indexes], color=colors[i])
            ax.plot(df["t"][indexes], df["dmin"][indexes], color=colors[i])
            ax.plot(df["t"][indexes], df["dmax"][indexes], color=colors[i])
            ax.fill_between(df["t"][indexes], df["dmin"][indexes], df["dmax"][indexes], color=colors[i], alpha=fillopacity, label=labels[i])
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 4)
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance to target (m)")

    ani = animation.FuncAnimation(fig, animate, frames=fps * duration, interval=1 / fps / 1000) 
    writer = animation.FFMpegWriter(fps=fps)

    def progress_callback(current_frame, total_frames):
        print(f"{current_frame}/{total_frames}")

    ani.save(f'/home/alex/Pictures/ScienceRoboticsPaper/Plot_{name_movie}.mp4', 
             writer=writer, 
             dpi=dpi, 
             progress_callback=progress_callback, 
             savefig_kwargs={"bbox_inches": 'tight'})
    print("Done")
    #fig.show()
