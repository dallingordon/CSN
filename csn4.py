import torch
import torch.nn as nn

##this one has the freq inverted correctly
class SineWaveLayer(nn.Module):
    def __init__(self, n, phi_prev=False, w_prev=False):
        super(SineWaveLayer, self).__init__()
        self.n = n
        self.phi_prev = phi_prev
        self.w_prev = w_prev
        
        # Parameters A, w, phi
        self.A = nn.Parameter(torch.rand(n, 1))
        self.w = nn.Parameter(torch.rand(n, 1))
        self.phi = nn.Parameter(torch.rand(n, 1))
        
        # Single phi_current parameter
        self.phi_current = nn.Parameter(torch.rand(1))
        
        if phi_prev:
            self.phi_previous = nn.Parameter(torch.rand(1))
        else:
            self.phi_previous = None
        
        # Single current_w parameter
        self.current_w = nn.Parameter(torch.rand(1))
        
        if w_prev:
            self.previous_w = nn.Parameter(torch.rand(1))
        else:
            self.previous_w = None

    def forward(self, t_vector, phi_previous=None, w_previous=None):
        # Expand t_vector to match the shape of parameters
        
        # Calculate the phase
        phi = self.phi
        phi_static = self.phi_current
        if self.phi_prev and phi_previous is not None:
            phi_static = phi_static + phi_previous
        phi = phi + phi_static
        
        # Calculate the angular frequency
        w = self.w
        w_static = self.current_w
        if self.w_prev and w_previous is not None:
            w_static = w_static + w_previous
        w = w + w_static
        
        # Calculate the sine wave
        sine_wave = self.A * torch.sin((1/w) * t_vector + phi)
        
        return sine_wave, phi_static , w_static

    def top_up_phi(self):
        with torch.no_grad():
            min_phi = self.phi.min()
            self.phi -= min_phi
            self.phi_current += min_phi
            
    def top_up_w(self):
        with torch.no_grad():
            min_w = self.w.min()
            self.w -= min_w
            self.current_w += min_w
    
class SelectorWaveLayer(nn.Module):
    def __init__(self, n, m, phi_prev=False, w_prev=False):
        super(SelectorWaveLayer, self).__init__()
        ###csn 4 has amplitudes as well.  
        self.n = n
        self.m = m
        self.phi_prev = phi_prev
        self.w_prev = w_prev
        
        # Parameters w, phi
        self.w = nn.Parameter(torch.rand(m, n))
        self.phi = nn.Parameter(torch.rand(m, n))
        self.a = nn.Parameter(torch.rand(m,n)) #new to csn4
        
        # Single phi_current parameter
        self.phi_current = nn.Parameter(torch.rand(1))
        
        if phi_prev:
            self.phi_previous = nn.Parameter(torch.rand(1))
        else:
            self.phi_previous = None
        
        # Single current_w parameter
        self.current_w = nn.Parameter(torch.rand(1))
        
        if w_prev:
            self.previous_w = nn.Parameter(torch.rand(1))
        else:
            self.previous_w = None

    def forward(self, t_vector, wave_input, phi_previous=None, w_previous=None):
        # Expand t_vector to match the shape of parameters
        
        phi = self.phi
        phi_static = self.phi_current
        if self.phi_prev and phi_previous is not None:
            phi_static = phi_static + phi_previous
        phi = (phi + phi_static).unsqueeze(-1)
        
        # Calculate the angular frequency
        w = self.w
        w_static = self.current_w
        if self.w_prev and w_previous is not None:
            w_static = w_static + w_previous
        w = (w + w_static).unsqueeze(-1)
        
        # Calculate the sine wave
        sine_wave = 0.5 * torch.sin((1/w) * t_vector + phi) + 0.5
        sine_wave = sine_wave * self.a.unsqueeze(-1)
        
        result = torch.einsum('ijk,jk->ik', sine_wave, wave_input)
        
        return result, phi_static, w_static
    
    def top_up_phi(self):
        with torch.no_grad():
            min_phi = self.phi.min()
            self.phi -= min_phi
            self.phi_current += min_phi
            ####this should likely do something with the frequency. only go to the positive w and if it exceeds reset to the negative, that way you are searching inside the frequency.  these local minima are good.  keep it stable and about 0
            
    def top_up_w(self):
        with torch.no_grad():
            min_w = self.w.min()
            self.w -= min_w
            self.current_w += min_w

class SineWaveNetwork(nn.Module):
    def __init__(self):
        super(SineWaveNetwork, self).__init__()
        
        # Layers
        self.sine_layer = SineWaveLayer(n=20)
        
        # Selector layers with increasing m
        self.selector_layer1 = SelectorWaveLayer(n=20, m=25)
        self.selector_layer2 = SelectorWaveLayer(n=25, m=30)
        self.selector_layer3 = SelectorWaveLayer(n=30, m=1)

    def forward(self, t_vector, initial_phi=None, initial_w=None):
        # First layer
        sine_output, phi, w = self.sine_layer(t_vector, initial_phi, initial_w)
        
        # Selector layers
        selector_output1, phi, w = self.selector_layer1(t_vector, sine_output, phi, w)
        selector_output2, phi, w = self.selector_layer2(t_vector, selector_output1, phi, w)
        selector_output3, phi, w = self.selector_layer3(t_vector, selector_output2, phi, w)
        
        return selector_output3

