from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
	def __init__(self, params, lr=1e-3):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		defaults = {"lr": lr}
		super().__init__(params, defaults)


	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()
		for group in self.param_groups:
			lr = group["lr"] # Get the learning rate.
			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p] # Get state associated with p.
				t = state.get("t", 0) # Get iteration number from the state, or 0.
				grad = p.grad.data # Get the gradient of loss with respect to p.
				p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
				state["t"] = t + 1 # Increment iteration number

		return loss


if __name__=="__main__":
	weights1 = torch.nn.Parameter(5 * torch.randn((10, 10)))
	weights2 = torch.nn.Parameter(weights1.clone())
	weights3 = torch.nn.Parameter(weights1.clone())
	print(weights1)
	print()


	print("\n")
	print("=" * 80)
	print("test1: lr=1e1")
	print("=" * 80)

	opt1 = SGD([weights1], lr=1e1)
	for t in range(10):
		opt1.zero_grad() # Reset the gradients for all learnable parameters.
		loss = (weights1**2).mean() # Compute a scalar loss value.
		print(loss.cpu().item())

		loss.backward() # Run backward pass, which computes gradients.
		opt1.step() # Run optimizer step.



	print("\n")
	print("=" * 80)
	print("test2: lr=1e2")
	print("=" * 80)

	opt2 = SGD([weights2], lr=1e2)
	for t in range(10):
		opt2.zero_grad() # Reset the gradients for all learnable parameters.
		loss = (weights2**2).mean() # Compute a scalar loss value.
		print(loss.cpu().item())

		loss.backward() # Run backward pass, which computes gradients.
		opt2.step() # Run optimizer step.


	print("\n")
	print("=" * 80)
	print("test2: lr=1e3")
	print("=" * 80)

	opt3 = SGD([weights3], lr=1e3)
	for t in range(10):
		opt3.zero_grad() # Reset the gradients for all learnable parameters.
		loss = (weights3**2).mean() # Compute a scalar loss value.
		print(loss.cpu().item())

		loss.backward() # Run backward pass, which computes gradients.
		opt3.step() # Run optimizer step.


'''
Parameter containing:
tensor([[  9.6722,  -3.9127,  -3.2207,   2.6492,   7.6735,   0.8575,  -2.9102,
          -3.7072,   0.6529,   4.1726],
        [ -2.1375,   0.9556,  -8.6788,   3.4573,   7.4729,   3.8625,   3.5115,
          -4.1830,  -1.8845,  -6.5021],
        [  6.9843,  -7.0412,  -0.9390,   6.8415,  -1.8587,   0.2172,  -2.9057,
           0.0895,  -0.8770,  -2.0599],
        [  2.3152,  -3.9575,   1.1606,  -2.3998,  -0.6742,  -0.3423,  -3.7691,
          -1.0950,   9.0428,  -3.8106],
        [  4.0556, -11.9883,   3.4431,   1.5960,  -0.1489,  -0.2961,  -3.3897,
           5.6967,   2.7860,  -0.4681],
        [  3.1078,  -2.7571,   7.4242,  -2.4447,  -3.8524,  -6.3584,  -1.8231,
          -1.6428,   1.3722,   4.5956],
        [  0.4258,  -7.1635, -11.4958,  -3.0108,  -0.2020,   9.3787,  -3.4653,
          -1.8681,   2.0156,  -7.7788],
        [  2.1674,   7.0790,  -1.2377,   3.7746,  -3.6996,  -3.6645,   3.8369,
          -1.9717,   7.4920,  -7.1944],
        [  0.7180,  -2.3578,   0.1886,  -9.5907,   1.1462,   3.1279,  -1.4286,
          -0.4399,  -3.5218,   6.3261],
        [ -1.0740,  -2.9192,   1.1768,  -0.4822,  -1.2615,  -0.0383,   1.6201,
          -8.2005,  -1.3026,  -1.1472]], requires_grad=True)



================================================================================
test1: lr=1e1
================================================================================
19.923667907714844
12.751148223876953
9.399600982666016
7.354184627532959
5.956889629364014
4.938943862915039
4.16534423828125
3.5594053268432617
3.073826313018799
2.6776442527770996


================================================================================
test2: lr=1e2
================================================================================
19.923667907714844
19.923667907714844
3.4183602333068848
0.08180905133485794
9.371193348828785e-17
1.0444776507482578e-18
3.517123496961457e-20
2.0951747804338352e-21
1.7973764060773417e-22
1.9970848430507638e-23


================================================================================
test2: lr=1e3
================================================================================
19.923667907714844
7192.44384765625
1242248.375
138186848.0
11193135104.0
706414903296.0
36265034514432.0
1560276927250432.0
5.750846963213926e+16
1.8466611149311836e+18

Learning Rate	Behavior
1e1	            Decays slowly
1e2	            Decays very fast
1e3	            Diverges / explodes

This is the classic learning-rate tradeoff:

Too small → slow learning
Good size → fast convergence
Too large → divergence / instability
'''