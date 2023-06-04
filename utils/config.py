sample_rate = 16000
stride_samples = sample_rate * 15
clip_samples = sample_rate * 30
kfold = 4

#labels = ['Cargo', 'Passengership', 'Tanker', 'Tug']
labels = ['Cargo', 'Dredger', 'Fishboat', 'Motorboat', 
          'Mussel boat', 'Natural ambient noise', 'Ocean liner', 'Passengers',
          'Passengership', 'Pilot ship', 'RORO', 'Sailboat',
          'Tanker', 'Trawler', 'Tug', 'Tugboat'
          ]
'''
ShipsEar = {
            'Fishboat':'A', 'Mussel boat':'A', 'Dredger':'A',
            'Motorboat':'B', 'Sailboat':'B',
            'Passengers':'C',
            'Ocean liner':'D', 'RORO':'D',
            'Natural ambient noise':'E',
           }
labels = ['A', 'B', 'C', 'D', 'E']

ShipsEar = {
            'Dredger':'A', 
            'Fishboat':'B', 
            'Motorboat':'C',
            'Mussel boat':'D', 
            'Natural ambient noise':'E',
            'Ocean liner':'F', 
            'Passengers':'G', 
            'RORO':'H',
            'Sailboat':'I',
           }
'''
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)

#------------------------------------------------------
mel_bins = 64
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None