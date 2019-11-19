import solver_single_temporal
if __name__ == '__main__':
	print "Training started"
	# solver_single_temporal.train()
	solver_single_temporal.validate()
	print "Training ended"
	print "testing started"
	print solver_single_temporal.test()
	print "testing ended"
