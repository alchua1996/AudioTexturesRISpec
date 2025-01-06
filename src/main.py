import torch
import torchaudio
from TextureSynthesisRISpec import *

torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    IN_PATH = '/content/drive/MyDrive/fire.wav'
    OUT_PATH = '/content/drive/MyDrive/syn_fire.wav'
    ITERS = 250
    #load in target sound texture
    waveform, sample_rate = torchaudio.load(IN_PATH)
    waveform = waveform.to(device)
    
    #initialize synthesized texture as WN/preprocess/put as variable
    noise = torch.randn_like(waveform)
    win_len = 512
    noise[:,:win_len // 2] = noise[:,:win_len // 2] * torch.hann_window(win_len, device = device)[:win_len // 2]
    noise[:,-win_len // 2:] = noise[:,-win_len // 2:] * torch.hann_window(win_len, device = device)[-win_len // 2:]
    syn_audio = torch.autograd.Variable(noise.clone().detach(),
                                        requires_grad=True).to(device)
    #class for making texture
    textureNet = TextureSynthesisRISpec(waveform, 128, device)
    #optimizer
    optimizer = torch.optim.LBFGS([syn_audio],
                                  line_search_fn = 'strong_wolfe',
                                  tolerance_grad = 0.0,
                                  tolerance_change = 0.0,
                                  )
    #training loop
    for k in range(250):
        def closure():
            optimizer.zero_grad()
            loss = textureNet(syn_audio)
            loss.backward()
            return loss
        optimizer.step(closure)
        if((k+1)%50 == 0):
            print('Epoch {} Complete!'.format(k+1))
            print('Loss is:', closure().item())
    
    #this needs to be done at the end or the texture will be noisy
    syn_audio = syn_audio * torch.std(waveform)/torch.std(syn_audio)
    torchaudio.save(OUT_PATH, syn_audio.detach().cpu(), sample_rate)