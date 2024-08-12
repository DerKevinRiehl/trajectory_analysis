

# EXAMPLE HOW TO RENDER PERFECTLY AS IMAGE AGAIN
from tools_video import getNumberOfFramesFromVideo, extractFrameFromVideo
import matplotlib.pyplot as plt
import numpy as np
import matplotlib



def makeNoiceFigure():
    video_file_path = "C:/VIDEO_ETH/DJI_0933.MOV"
    a =  getNumberOfFramesFromVideo(video_file_path)
    a,b = extractFrameFromVideo(video_file_path, 0)
    # last = matplotlib.get_backend()
    # matplotlib.use('Agg') # make a user that does not exist so window doesnt popup
    plt.ioff()
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(b.shape[1]/100,b.shape[0]/100)
    
    # variant 1
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_xlim(0, b.shape[1])
    ax.set_ylim(b.shape[0], 0)
    fig.add_axes(ax)
    
    # variant 2
    # plt.axis('off')

    ax.imshow(b, aspect='auto')


    ax.plot([-1000,1000], [0,1000], "red", linewidth=15)

    ax.scatter(200.5, 200.5, s=10, color="red")

    # SAVE TO FILE
    fig.savefig('out.jpg',  bbox_inches='tight', pad_inches=0, dpi=100)

    print(fig.canvas.get_width_height())
    # EXAMPLE HOW TO GET IMAGE as NUMPY ARRAY again
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # matplotlib.use(last)
    plt.ion()
    return data

data = makeNoiceFigure()
plt.figure()
plt.imshow(data)