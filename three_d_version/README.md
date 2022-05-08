# 1. PyBullet Hinge

The definition of "BulletHingeConstraint using frames" is not easy to find, a good way to understand it is by reading [the implementation of "getHingeAngle()"](https://github.com/bulletphysics/bullet3/blob/old_demos/src/BulletDynamics/ConstraintSolver/btHingeConstraint.cpp). The relevant C++ codesare pasted here in case the link is invalidated.
```cpp
btScalar btHingeConstraint::getHingeAngle()
{
	return getHingeAngle(m_rbA.getCenterOfMassTransform(),m_rbB.getCenterOfMassTransform());
}

btScalar btHingeConstraint::getHingeAngle(const btTransform& transA,const btTransform& transB)
{
	const btVector3 refAxis0  = transA.getBasis() * m_rbAFrame.getBasis().getColumn(0);
	const btVector3 refAxis1  = transA.getBasis() * m_rbAFrame.getBasis().getColumn(1);
	const btVector3 swingAxis = transB.getBasis() * m_rbBFrame.getBasis().getColumn(1);
	btScalar angle = btAtan2(swingAxis.dot(refAxis0), swingAxis.dot(refAxis1));
	return m_referenceSign * angle;
}
```

To understand this function, a lemma should be introduced first.
```
If "M" is a matrix of rotation, e.g. the Yaw-Pitch-Roll matrix, then "M.getColumn(0)" is the "direction vector of the rotated x-axis". 
```

It's easy to prove this lemma by considering "M.getColumn(0) == multiply(M, transpose(1, 0, 0))". Same applies for "M.getColumn(1)" and "M.getColumn(2)". 
Now it's obvious that "hinge angle" here is defined as the angle between "rotated-y-axis-of-frameB" and "rotated-y-axis-of-frameA", and **it's assumed that "rotated-y-axis-of-frameB(swingAxis) is perpendicular to rotated-z-axis-of-frameA(ref frame, containing refAxis0 & refAxis1)"**, although the formula can calculate a result without this assumption it'd be quite difficult to imagine the picture in that case, i.e. unrealistic use case.

# 2. Eigenvector rotation

A few keys are prepared for rotating the cube w.r.t. x-, y-, z-axis in its own body frame, then a single key is prepared for recovering the orientation by a single act along the eigen vector.

# 3. Blender and blend2bam
Please use `Blender 3.0.1`, `panda 1.10.1` and `blend2bam 0.20` to edit the blender and bam files in `models` directory. Use the following command to refresh the displaying rocket bam file.
```
proj-root/three_d_version/models> blend2bam rocket_with_engine.blend rocket.bam
```

# 2. Multi-episode memory leak debugging run
```
proj-root/three_d_version> python3 lunarlander_offline_gfold.py --ep=20 --debugleak=1 
```

