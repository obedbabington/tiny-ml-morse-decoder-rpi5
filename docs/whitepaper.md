---
title: "MorseAI: An End-to-End Edge AI Workflow on Raspberry Pi 5"
subtitle: "A beginner-friendly tutorial for building and deploying neural networks on Arm-based edge hardware"
author: "Obed Babington"
date: "______________"
---

> **Source note.** This Markdown is the editable source for the review document.
> The formatted Word version is `docs/MorseAI_EdgeAI_RaspberryPi5.docx`
> (regenerate with `PYTHONPATH=.build_tools python3 docs/build_docx.py`).
> If you prefer pandoc: `pandoc docs/whitepaper.md -o MorseAI.docx`.

![Figure 1. The MorseAI Edge AI pipeline running on a Raspberry Pi 5.](architecture.png)

---

## Executive Summary

MorseAI is a compact, complete example of an Edge AI application: a user taps a
letter in Morse code on a single push button, and a neural network running
entirely on a Raspberry Pi 5 identifies the letter in real time. There is no
cloud service and no network round-trip: capture, preprocessing, and inference
all happen on the device.

The project began as an Arm workshop on an STM32 microcontroller and was ported
to the Raspberry Pi 5 to serve as an accessible teaching platform. Its value is
not the Morse problem itself but the workflow it demonstrates: capturing a sensor
signal, turning it into features, training a small model off-device, converting
it to TensorFlow Lite, and running inference on Arm hardware. That same
four-stage pattern underpins practical edge workloads such as sensor
classification, vibration-based predictive maintenance, gesture recognition, and
anomaly detection.

The deployed model is approximately 4.2 KB and imposes negligible load on the Pi
5, leaving ample headroom to scale up to real sensors and larger models. This
document describes the motivation, architecture, implementation decisions,
measured resource cost, and the paths from this tutorial to production Edge AI
systems, including a comparison with the original microcontroller
implementation.

## 1. Introduction and Motivation

Most introductions to machine learning stop at a notebook: a model is trained, an
accuracy number is reported, and the work ends there. The harder and more useful
skill is getting a model off the laptop and onto hardware where it has to deal
with real, messy input. MorseAI was built to teach exactly that transition, using
the smallest honest example we could find.

Morse code is a good first signal to classify because each character is just a
handful of button presses that differ only in duration. The data is tiny and
human-readable, you can generate your own dataset in minutes, and the problem is
genuinely temporal: the same class of concern (timing, noise, sampling) that
appears in audio and vibration, but slow enough to debug by hand. It is
explicitly a learning example, not an industrial product.

## 2. Why Morse Code Is a Good First Edge AI Problem

- **Small, interpretable input.** Each character is at most four press durations:
  four numbers you can read and sanity-check by eye.
- **Genuinely temporal.** A dot and a dash differ only in duration, which forces
  real thinking about timing and noise.
- **You own the data.** A usable dataset is collected by pressing a button, so the
  learner controls the whole pipeline end to end.
- **Instructive failure modes.** Ambiguous taps show why a learned model is more
  forgiving than a hand-tuned threshold, motivating the use of ML in the first
  place.

## 3. System Architecture

The system is a four-stage pipeline. Each stage maps to a single source file,
which keeps every part independently testable. The training step runs once,
off-device; everything in Figure 1 runs on the Pi.

**Table 1. Mapping of pipeline stages to source files.**

| Stage | File | Responsibility |
|-------|------|----------------|
| Signal capture | `capture.py` | Time button presses, debounce, detect end-of-character |
| Preprocessing | `inference.py` | Pad to four values, apply the training StandardScaler |
| Inference | `inference.py` | Run the TFLite interpreter; return top class + confidence |
| Output | `main.py` | Wire capture to inference; print results; hold session state |

## 4. Implementation

### 4.1 Signal Capture

The button is read with gpiozero, which fires callbacks on press and release. A
monotonic, high-resolution clock (`perf_counter_ns`) measures each press duration
in microseconds. Two details matter more than they appear: a 50 ms debounce
filter, so mechanical contact chatter registers as a single press; and a 2-second
end-of-character timeout, implemented as a cancel-and-rearm timer that decides
when a character is complete. Each character is stored as up to four durations and
zero-padded to a fixed length of four.

### 4.2 Preprocessing and Feature Representation

Raw press durations span tens to hundreds of thousands of microseconds. A
StandardScaler (subtract mean, divide by standard deviation) brings them into a
range the network trains on efficiently. Critically, the identical transform must
be applied at inference time, so the scaler parameters are exported with the model
and reloaded on the device. A normalization mismatch is the most common cause of a
model that loads correctly but predicts nonsense.

### 4.3 Model Design and Training

The classifier is a small feed-forward network: two hidden layers of 16 ReLU
units feeding an 11-way softmax (ten letters A–J plus one "unclassified" class).
With only four inputs and ten classes, depth is unnecessary and a larger model
would simply overfit the 450-sample dataset. The eleventh class is trained on a
small fraction of random noise vectors so the model can respond "I don't know"
rather than forcing garbage onto the nearest letter, a small change that makes
the deployed system noticeably more robust. Training uses Adam, sparse categorical
cross-entropy, and early stopping on validation loss.

### 4.4 Conversion to TensorFlow Lite

The trained Keras model is converted with the TFLite converter using default
optimizations (weight quantization). Conversion produces two artifacts: the model
file (4,276 bytes) and a normalization configuration file holding the scaler mean
and scale. Shipping the normalization parameters alongside the model is what makes
the result reproducible on a different machine.

### 4.5 On-Device Inference

Inference on the Pi uses `tflite-runtime` rather than full TensorFlow: it is a few
megabytes instead of hundreds, starts quickly, and is sufficient for running (not
training) a model. A prediction normalizes the input, sets the input tensor,
invokes the interpreter, and takes the argmax of the softmax output, returning the
predicted letter with a confidence score and a measured inference time.

## 5. Performance on Raspberry Pi 5

Edge AI is judged on resource cost. The figures below were measured on a
**Raspberry Pi 5** with `python benchmark.py` (10,000 iterations, 4 threads,
200 warmup). TFLite loads the **XNNPACK** CPU delegate automatically.

**Table 2. Measured resource cost of the MorseAI model (Raspberry Pi 5).**

| Metric | Value | How measured |
|--------|-------|--------------|
| Model size (disk) | **4.18 KB** (4,276 bytes) | `ls -l` of the `.tflite` file |
| Input / output | 4 × float32 → 11 × float32 | model signature |
| Inference latency (median) | **3.0 µs** | warm `invoke()` calls |
| Inference latency (mean / p95) | **3.1 / 3.1 µs** | same run |
| Inference latency (min / max) | **3.0 / 79.8 µs** | same run |
| Throughput | **~280,400 inferences/s** | tight loop, 4 threads |
| Interpreter memory (RSS Δ) | **2.7 MiB** | 39.5 MiB before load → 42.2 MiB after |
| Process RSS after load | **42.2 MiB** | `/proc` resident set size |
| CPU during benchmark loop | **100%** | process CPU time ÷ wall time |

Latency is far below one millisecond. In normal use with `main.py`, the bottleneck
is the 2-second character timeout, not inference.

**CPU note:** The benchmark uses a tight loop with no idle time between calls.
100% means the process kept one core busy for the timed section. It does not mean
all four Pi cores were saturated, and it is not typical of CPU use while waiting
for button input.

## 6. Extending the Pattern to Real-World Workloads

The pipeline does not change when the problem becomes serious. Replace the button
with a different sensor, retrain on the new data, and redeploy the same way:
capture → preprocess → TFLite inference → act.

**Table 3. The same pipeline applied to practical Edge AI workloads.**

| Application | What changes | What stays the same |
|-------------|--------------|---------------------|
| Sensor classification | Accelerometer / environmental sensor instead of a button | Fixed-length feature vector, small MLP |
| Predictive maintenance / vibration | Windowed vibration signal; normal vs. fault label | Time-series in, classification out |
| Gesture recognition | IMU data from a wearable | Variable-length input → fixed feature trick |
| Anomaly detection | Learn "normal"; flag drift | The "unclassified" idea, generalised |
| Other IoT edge workloads | Keyword spotting, occupancy, fault detection | On-device decision, no cloud round-trip |

The step up from this tutorial is usually more and better data, a slightly larger
model, and windowing a continuous stream rather than discrete presses; none of
which change the deployment story.

## 7. STM32 vs. Raspberry Pi 5: Two Faces of Edge AI

Because the project exists on both an STM32 microcontroller and a Raspberry Pi 5,
it offers a clear side-by-side of two points on the Edge AI spectrum. Neither is
strictly better; they answer different questions.

**Table 4. Developer-facing comparison of the two implementations.**

| Dimension | STM32 (microcontroller) | Raspberry Pi 5 (Linux SBC) |
|-----------|-------------------------|----------------------------|
| Developer experience | Bare-metal / RTOS, C/C++, flash-and-debug | Full Linux, Python, edit-and-run |
| Toolchain | TFLite Micro, C compiler, model as C array | TensorFlow → TFLite, tflite-runtime, pip/venv |
| Deployment workflow | Cross-compile and flash firmware | `git pull` and run a script |
| Compute & memory | Hundreds of KB RAM, MHz core, no OS | GBs of RAM, GHz quad Cortex-A76, full OS |
| Iteration speed | Slow: recompile + reflash | Fast: edit a line, rerun in seconds |
| Power & cost | Milliwatts; cents to dollars | Watts; tens of dollars |
| Best for | Always-on, battery, cost-sensitive, real-time | Prototyping, heavier models, learning, gateways |

A realistic workflow uses both: develop and validate the model on the Pi, where
the fast edit-run loop keeps attention on the problem, then port the proven model
down to the STM32 with TFLite Micro when microwatt power, cents-level unit cost,
or hard real-time guarantees are required. The Pi is the workshop; the
microcontroller is the finished product.

## 8. Challenges and Lessons Learned

- **The signal is harder than the network.** Debounce, the end-of-character
  timeout, and padding took more effort than the model, typical for Edge AI.
- **Normalization is a deployment concern.** Shipping the scaler with the model is
  essential; a mismatch yields confident nonsense.
- **An explicit "I don't know" class pays off.** It lets the system degrade
  gracefully on ambiguous input.
- **`tflite-runtime` vs. TensorFlow is a real choice.** The runtime is the right
  tool for inference-only deployment and shaped the Python setup.
- **Threads don't always help tiny models.** Per-call overhead can dominate;
  `benchmark.py` makes it easy to compare 1, 2, and 4 threads.

## 9. Conclusion and Future Work

MorseAI shows that a complete Edge AI workflow, from sensor capture to on-device
inference, fits in a few small, readable files on a Raspberry Pi 5. The Morse
problem is intentionally simple so that the workflow, which transfers directly to
practical sensor and IoT applications, stays in focus. For a beginner, it is a
realistic first deployment; for a practitioner, it is a clean template to adapt.

Future work:

- Grow and balance the dataset; extend coverage to full A–Z and digits.
- Decode whole words using inter-character and inter-word gaps.
- Add an int8-quantized variant and compare accuracy against size.
- Provide TFLite Micro build notes for porting back to the STM32 target.

## Appendix A: Hardware Setup and Wiring

**Table A1. Button wiring.** The code enables the internal pull-up, so no external
resistor is required.

| Button terminal | Pi physical pin | Pi function | Role |
|-----------------|-----------------|-------------|------|
| Terminal 1 | Pin 11 | GPIO 17 (BCM) | Signal in |
| Terminal 2 | Pin 9 | GND | Ground |

## Appendix B: Reproducing the Results

- Train and convert the model with `training/MorseAI_RPi_Notebook.ipynb`.
- Deploy on the Pi with `setup_pi.sh`, then run `python main.py`.
- Measure performance on the Pi with `python benchmark.py`.
- Regenerate the architecture figure with `docs/make_diagram.py`.
