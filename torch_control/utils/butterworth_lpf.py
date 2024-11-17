import torch
import numpy as np
import math
from scipy.signal import butter

class ButterworthLowPassFilter:
    def __init__(self, cutoff_frequency, dt, order=1):
        """
        Butterworth low-pass filter designed for real-time filtering in robotic control.
        This implementation maintains internal state for continuous sample-by-sample processing.
        :param cutoff_frequency: Cutoff frequency in Hz.
        :param dt: Sampling interval in seconds.
        :param order: Filter order. Only 1 and 2 are supported.
        """
        self.cutoff_frequency = cutoff_frequency
        self.dt = dt
        self.order = order

        nyquist = 0.5 / dt
        normalized_cutoff = cutoff_frequency / nyquist
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        
        # Convert to tensor and store coefficients
        self.b = torch.tensor(b, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)

        # Initialize state vectors for previous inputs and outputs (length determined by filter order)
        self.x_prev = torch.zeros(len(self.b) - 1, dtype=torch.float32)  # Previous inputs
        self.y_prev = torch.zeros(len(self.a) - 1, dtype=torch.float32)  # Previous outputs

    def filter(self, x):
        """
        Process a single sample through the filter and update the internal state.
        :param x: A single input sample.
        :return: A single filtered output sample.
        """
        x = torch.tensor([x], dtype=torch.float32)  # Ensure input is tensor for consistency
        y = self.b[0] * x  # Current output initialized with current input contribution

        # Add contributions from previous inputs and outputs
        for i in range(len(self.x_prev)):
            y += self.b[i + 1] * self.x_prev[i]
        for j in range(len(self.y_prev)):
            y -= self.a[j + 1] * self.y_prev[j]

        y /= self.a[0]  # Normalize by a[0] if it's not equal to 1

        # Update internal states
        if len(self.x_prev) > 0:
            self.x_prev = torch.cat((x, self.x_prev[:-1]))
        if len(self.y_prev) > 0:
            self.y_prev = torch.cat((y, self.y_prev[:-1]))

        return y.item()  # Return as Python scalar for convenience

class ButterworthLowPassFilterBatchedParallel:
    def __init__(self, cutoff_frequency, dt, batch_size, action_size, order=1, device='cpu'):
        """
        Butterworth low-pass filter designed for batched real-time filtering in robotic control,
        utilizing PyTorch parallelism for efficient batch processing, with separate state maintenance for each sample.
        :param cutoff_frequency: Cutoff frequency in Hz.
        :param dt: Sampling interval in seconds.
        :param batch_size: The number of samples in each batch.
        :param action_size: The size of each action/command in the batch.
        :param order: Filter order. Only 1 and 2 are supported.
        """
        self.cutoff_frequency = cutoff_frequency
        self.dt = dt
        self.order = order
        self.batch_size = batch_size
        self.action_size = action_size

        nyquist = 0.5 / dt
        normalized_cutoff = cutoff_frequency / nyquist
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        
        self.b = torch.tensor(b, dtype=torch.float32, device=device)
        self.a = torch.tensor(a, dtype=torch.float32, device=device)

        # Initialize state vectors for each sample and each action dimension
        self.x_prev = torch.zeros(batch_size, action_size, len(b) - 1, dtype=torch.float32, device=device)
        self.y_prev = torch.zeros(batch_size, action_size, len(a) - 1, dtype=torch.float32, device=device)

    def reset(self, idx=None):
        """
        Reset the internal state of the filter.
        :param idx: The index of the sample to reset. If None, reset all samples.
        """
        if idx is None:
            self.x_prev = torch.zeros_like(self.x_prev)
            self.y_prev = torch.zeros_like(self.y_prev)
        else:
            self.x_prev[idx] = torch.zeros_like(self.x_prev[idx])
            self.y_prev[idx] = torch.zeros_like(self.y_prev[idx])

    def filter(self, x):
        """
        Process a batch of samples through the filter in parallel and update the internal state.
        :param x: A batch of input samples with shape (batch_size, action_size).
        :return: A batch of filtered output samples with the same shape as x.
        """
        # Expand filter coefficients to match batch and action dimensions for broadcasting
        # b_expanded = self.b.unsqueeze(0).unsqueeze(0)
        # a_expanded = self.a.unsqueeze(0).unsqueeze(0)

        # Current output contribution
        y = x * self.b[0] #b_expanded[..., 0]

        # Vectorized contributions from previous inputs
        xb = self.x_prev * self.b[1:] #b_expanded[..., 1:]
        xb_sum = torch.sum(xb, dim=2)

        # Vectorized contributions from previous outputs
        yb = self.y_prev * self.a[1:] #a_expanded[..., 1:]
        yb_sum = torch.sum(yb, dim=2)

        # Combine contributions
        y = (y + xb_sum - yb_sum) / self.a[0] #a_expanded[..., 0]

        # Update states
        self.x_prev = torch.roll(self.x_prev, shifts=1, dims=2)
        self.x_prev[..., 0] = x
        self.y_prev = torch.roll(self.y_prev, shifts=1, dims=2)
        self.y_prev[..., 0] = y

        return y

def plot_frequency_response(ax, signal, fs, title='Frequency Response'):
    """
    Plot the magnitude spectrum of a signal.
    """
    # Compute FFT
    signal_fft = torch.fft.fft(signal)
    # Compute frequency bins
    freq = torch.fft.fftfreq(len(signal), 1/fs)
    
    # Plot magnitude spectrum in log scale
    # ax.plot(freq, torch.abs(signal_fft), label=title)
    ax.semilogy(freq[1:-5], torch.abs(signal_fft[1:-5]), label=title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_xlim(0, fs/2)  # Nyquist limit
    ax.grid(True)
    ax.legend()

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 33.0  # Sampling frequency in Hz
    dt = 1.0 / fs  # Sampling interval
    cutoff = 4  # Cutoff frequency in Hz
    
    # Create filter instances
    # lp_filter_first_order = ButterworthLowPassFilter(cutoff, dt, order=1)
    # lp_filter_second_order = ButterworthLowPassFilter(cutoff, dt, order=2)
    lp_filter_first_order = ButterworthLowPassFilterBatchedParallel(cutoff, dt, batch_size=5, action_size=1, order=1)
    lp_filter_second_order = ButterworthLowPassFilterBatchedParallel(cutoff, dt, batch_size=5, action_size=1, order=2)
    
    # Generate a test signal: sum of 1 Hz and 10 Hz sine waves with noise
    t = torch.linspace(0, 1, int(fs), dtype=torch.float32)
    # signal = torch.sin(2 * np.pi * 1 * t) + 0.5 * torch.sin(2 * np.pi * 10 * t) + 0.1 * torch.sin(2 * np.pi * 100 * t) + 0.1 * torch.randn_like(t)
    # signal = signal.unsqueeze(0)  # Add batch dimension
    signal = []
    for i in range(5):
        signal.append(torch.sin(2 * np.pi * i * t) + 0.5 * torch.sin(2 * np.pi * 10 * t) + 0.1 * torch.sin(2 * np.pi * 100 * t) + 0.1 * i * torch.randn_like(t))
    signal = torch.stack(signal, dim=0)
    
    # Filter the signal
    filtered_signal_first_order = torch.zeros_like(signal)
    filtered_signal_second_order = torch.zeros_like(signal)
    for i in range(signal.shape[1]):
        filtered_signal_first_order[:, i] = lp_filter_first_order.filter(signal[:, i].unsqueeze(1)).squeeze(1)
        filtered_signal_second_order[:, i] = lp_filter_second_order.filter(signal[:, i].unsqueeze(1)).squeeze(1)
    # filtered_signal_first_order = lp_filter_first_order.filter(signal)
    # filtered_signal_second_order = lp_filter_second_order.filter(signal)
    
    batch_idx = 1

    # Plotting in time domain
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, signal[batch_idx], label='Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, filtered_signal_first_order[batch_idx], label='First-Order Filtered Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, filtered_signal_second_order[batch_idx], label='Second-Order Filtered Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.title('Time Domain')
    plt.savefig('time_domain.png')

    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    # Plotting in frequency domain
    plot_frequency_response(axs[0], signal[batch_idx], fs, title='Original Signal')
    plot_frequency_response(axs[1], filtered_signal_first_order[batch_idx], fs, title='First-Order Filtered Signal')
    plot_frequency_response(axs[2], filtered_signal_second_order[batch_idx], fs, title='Second-Order Filtered Signal')
    fig.suptitle('Frequency Domain Comparison')
    fig.tight_layout()
    fig.savefig('frequency_domain.png')