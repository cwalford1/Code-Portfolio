import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

"""About this task:
Here I'm plotting some sound data I collected for an experiment exploring some analogies between acoustics and quantum mechanics.  
The data represents the amplitude of the standing in a cavity produced by a speaker playing various frequencies.
Each plot corresponds to the different sizes of holes in an aluminum disk placed within the wave cavity. 
This produces a pair of coupled wave cavities which can be explored mathematically to reveal that even ordered resonance frequencies are unaffected by the coupling while odd ordered frequencies should shift. 
The goal here is to observe a shift in location of odd ordered resonance frequencies (translation: the relative locations of half the peaks should drift with the changing disk size)
I think these plots are interesting because they allow one to visually confirm an esoteric mathematical result:
the even order resonance frequencies (for example the large peaks in the center of each graph) don't shift while odd ordered one's (such as the peak immediately after) do.
Note: the even ordered peaks do shift a little but this is because we couldn't manufacture a perfectly symmetic wave cavity."""

#file names and information for each file
file1='./sameside/drive-download-20200225T185547Z-001/No Plate.xlsx'
file2='./sameside/drive-download-20200225T185547Z-001/1.184_ ID Plate.xlsx'
file3='./sameside/drive-download-20200225T185547Z-001/0.992_ ID Plate.xlsx'
file4='./sameside/drive-download-20200225T185547Z-001/0.830_ ID Plate.xlsx'
file5='./sameside/drive-download-20200225T185547Z-001/0.471_ ID Plate.xlsx'
file6='./sameside/drive-download-20200225T185547Z-001/0.192_ ID Plate.xlsx'
file7='./sameside/drive-download-20200225T185547Z-001/0_ ID Plate.xlsx'
colors=['blue','black','red','green','grey','magenta','cyan']
labels=['No Disk','1.184 (in) Hole','0.992 (in) Hole','0.830 (in) Hole','0.471 (in) Hole','0.192 (in) Hole','Solid Disk']
files=[file1,file2,file3,file4,file5,file6,file7]

mode='all'
#create plots
fig,axs=plt.subplots(len(files))
plt.subplots_adjust(hspace=1)
fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', va='center',fontsize='xx-large')
fig.text(0.5,0.9,'Resonance Frequencies of Perterbed Wave Cavity',ha='center',va='center',fontsize='xx-large')
fig.text(0.06, 0.5, 'Normalized Amplitude (arb units)', ha='center', va='center', rotation='vertical',fontsize='xx-large')

if mode=='all':
    for i,file in enumerate(files):
        #open data and save to numpy arrays
        df = pd.read_excel (r'{}'.format(file))
        xframe=pd.DataFrame(df,columns=['Untitled'])
        yframe=pd.DataFrame(df,columns=['Untitled 1'])
        x=xframe.to_numpy()
        y=yframe.to_numpy()
        #normalize data and redefine zero potential
        y*=1/max(y)
        y-=min(y)
        #plot each curve
        axs[i].plot(x,y,color=colors[i],label=labels[i])
        axs[i].legend(loc='upper right',fontsize='small')
        
plt.show()
