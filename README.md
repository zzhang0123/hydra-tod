# MeerKAT.hydra-tod


## Data Structures

### Raw data:
{(scan, receiver): ndarray[time, frequency]}


### Chunked data:
{(scan, receiver, t_chunk, f_chunk): ndarray[time, frequency]}
We chunk a list of times into pieces based on a given time scale. Also chunk the list of frequencies in a similar way.

### Mean gain parameters
(scan, receiver, t_chunk, f_chunk)
For each chunk, each scan and each receiver, we parameterise the average gain as an independent parameter.

### Noise parameters
(scan, receiver, 4)
For each scan and each receiver, we use four parameters {f_0, \alpha, \beta, \xi} to characterise the 1/f noise. In other words, we assume stationary 1/f noise statistics for the receiver over the whole scan.

### Sky parameters
ndarray[pixel, frequency]

from sky space to data space: linear_op(Tsky):= proj_Tsky @ Tsky  
==>    proj_Tsky: ndarray, shape (N<sub>time</sub>, N<sub>pix</sub>) for beam-convolved Tsky
                           shape (N<sub>time</sub>, N<sub>pix</sub>, N<sub>freq</sub>) for deconvolving T<sub>sky</sub> with the beam.

