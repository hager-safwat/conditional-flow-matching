# Conditional flow matching from scratch (2-D)

A minimal, self-contained implementation of **flow matching** (Lipman et al.,
2023), trained with the conditional flow-matching objective and **label
conditioning** so you can steer *which* part of the distribution is
generated. Runs on CPU in under a minute.

The target is a two-moons distribution where each moon is a class
(0 = upper, 1 = lower). After training, asking for class 0 generates only the
upper moon and class 1 only the lower one — a *controllable* generator.

![result](figure.png)

## How it differs from diffusion

A diffusion model learns to **denoise** in many small steps. Flow matching
instead learns a **velocity field** that transports a simple base
distribution (a Gaussian) into the data along near-straight paths, and
generates by integrating an ODE. It is often simpler and needs far fewer
sampling steps.

## The two ideas

**Straight-line conditional paths.** Pick a noise point `x0 ~ N(0, I)` and a
real point `x1`. Interpolate linearly:

```
x_t = (1 - t) * x0 + t * x1,      t in [0, 1]
```

Along that straight path the ideal velocity is constant: `u = x1 - x0`.

**Regress the velocity.** Train a network `v_theta(x_t, t, c)` — also given
the class label `c` — to predict `u`. That single MSE objective is the whole
of conditional flow matching. To generate a sample of class `c`, start from
noise and integrate `dx/dt = v_theta(x, t, c)` from `t = 0` to `t = 1`.

## What the figure shows

- **Left:** the target — two moons, one class each.
- **Middle:** conditional generation — class 0 and class 1 requested
  separately, each producing the correct moon.
- **Right:** the transport paths — noise at `t = 0` flowing along the learned
  velocity field to the requested class at `t = 1`.

Quantitatively, class-0 samples land ~0.02 from the real upper moon and ~0.81
from the lower one (and vice versa) — so conditioning genuinely controls which
mode is generated, rather than just reproducing the whole distribution.

## Run it

```bash
pip install -r requirements.txt
python cfm_demo.py
```

CPU-only, under a minute. It prints the loss, the conditioning check, and
regenerates `figure.png`.

## Scope and honesty

This is a **mechanism demo on synthetic data** — the smallest complete
conditional-flow-matching setup that still shows every moving part. It is not
a benchmark. The technique comes from my master's coursework, where I applied
conditional flow matching to an image-based problem; this repository
reimplements the core method from scratch on toy data, with no project data
involved.

## Reference

- Y. Lipman, R. Chen, H. Ben-Hamu, M. Nickel, M. Le, *Flow Matching for
  Generative Modeling*, ICLR 2023.

## License

MIT — see [LICENSE](LICENSE).
