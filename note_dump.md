If compressed computation is not computation in superposition, what is?

A toy model was given in a paper by Dan Braun where they claim computation in superposition must be happening (extract of the paper in this directory). Later, there was a LessWrong post (also in this directory) showing that this toy model did not really have computation in superposition by partially reverse engineering the algorithm. I would like to see if we can come up with a toy model that we believe does exhibit computation in superposition. As a first step, we should try and replicate the results of the LessWrong post (this is ongoing, don't worry about it)

I think this comment by Lucius is important:

"This investigation updated me more toward thinking that Computation in Superposition is unlikely to train in this kind of setup, because it's mainly concerned with minimising _worst-case_ noise. It does lots of things, but it does them all with low precision. A task where the model is scored on how close to correct it gets many continuously-valued labels, as scored by MSE loss, is not good for this.  
  
I think we need a task where the labels are somehow more discrete, or the loss function punishes outlier errors more, or the computation has multiple steps, where later steps in the computation depend on lots of intermediary results computed to low precision."

I think a natural thing to try is to make outlier errors punished more harshly - that should be the smallest change from what is currently done, as it does not involve coming up with a new task. Instead of having L^2 loss we can do L^n loss, with n >> 2. We could something like n=10?

There's of course the question of whether computation in superposition is possible in this setup at all. Should be? Or, better put, is there a reason why it wouldn't be?

We are interested in the case in which there is no mixing matrix.  What is a good level of sparsity? Not sure! 0.01 feels good but that is vibe-based. Hm in that case we expect one feature active per input (100 features, 50 neurons) - is that good? bad? Should we bump up the number of features? Like factor of 10 larger than number of neurons? Let's try and reason through this!

I guess Lucius' point hints at it being better for this purpose to have discrete labels rather than a continuous approximation. Could we just bin the output (e.g., maybe 20 bins from -1 to 1) and give pass/fail according to whether the model bins correctly? I suppose that would punish failure more harshly and encourage being quite good at approximating all features.

There's two knobs we can play with regarding what the problem even is:
1. Number of total features (wrt neurons; ignore)
	1. The overcompleteness ratio grows, so the naive solution must ignore a larger fraction of the total amount of features - that seems just strictly worse for it vs CiS solutions?
	2. Hm it seems that for the same amount of features active in expectation, the combination of two parameters (total number of features and prob of active) that has higher total number of features is worse for the naive solution, while not making CiS harder?
		1. This is roughly right yeah
2. Sparsity (probability of a feature being active)
	1. As probability of a feature being active increases, more of the ignored features get punished
	2. However, the more features are active at a given time, the more interference there is - ideally, a given set of active features must be approximately orthogonal so as to minimize interference and enable computation in superposition
	3. So if there's too few active features, the model does not get punished much for ignoring features, and hence there's no need to learn superposition
	4. If there's too many, computation in superposition suffers from too much interference and is not feasible
	5. In principle the 'too few' point shouldn't matter so much in terms of what's optimal? hm maybe that's not true, because there's a trade-off with not ignoring, which is that you don't learn the others perfectly
		1. There's also the question of: even if CiS is optimal, will gradient descent find it? If p is very low, then the gradient signal is super low/sparse - so might just get stuck at something approximating naive baseline
	6. is it right to assume that there's a sweet spot here? p too low and might as well ignore because any given feature does not occur often enough to matter; p too high and computation in superposition falls to superposition

Ok let's say we go for 50 neurons, 500 features, p=0.02. This gives 10 features active in expectation. That's plenty, but quite a bit less than number of neurons.

So... what about the loss? What type of loss is better? Is there a reason L^2 has become the de facto standard? Are we running into issues by reinventing the wheel here?

Let's try L^2 and L^4 for the loss

How will we find signatures of CiS? Well I guess just compare loss with the naive loss - if we do better, we can be pretty convinced we've found CiS, right? Next step is to actually inspect the solution, but let's not get ahead of ourselves.

Well, I think we have CiS with L^4 loss?
![[cis_l2_vs_l4_comparison.png]]
The loss per feature is the same for all features in the L4 trained solution, and they all do better than the 'ignore' solution (which is the red line). The L2 solution looks like the naive baseline-ish - ~50 features are done better than 'ignoring', a majority is terrible. Very asymmetric.

I think a next step would be to go and actually try to figure out what the model is actually doing in the L4 case.