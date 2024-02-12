The files in this directory contain gene expression and enriched pathway information for different types of patient trajectories.
Types of trajectories signifying the transition from one cancer stage to another: 
	I_to_I, 
	I_to_II, 
	II_to_II, 
	II_to_III, 
	III_to_III

All trajectories are of length 50 timesteps.
For each timestep the files contain data on 29 pathways and 8954 gene expressions.
Pathways are binary - 0 (normally expressed) or 1 (over-expressed).
Gene expressions are some type of float.

Some information on the amount and types of trajectories in each file:

			ductal_no_negatives.csv		lobular_w_negatives.csv
	I_to_I, 		   0				1150
	I_to_II,		 7000				1150
	II_to_II, 		   0				5500
	II_to_III, 		 22300				5500
	III_to_III		   0				5100
