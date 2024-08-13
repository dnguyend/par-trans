The numpy version. While this version does not offer the AD feature of JAX, it  may be faster on CPU, and we can use expm_multiply, which is not yet implemented in JAX

par\_trans.manifolds
====================

.. automodule:: par_trans.manifolds

	Stiefel
	========
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: par_trans.manifolds.stiefel
	.. autofunction:: par_bal
	.. autofunction:: solve_w
	.. autoclass:: Stiefel		       
	   :members:
   
        Flag
	=====
	.. autosummary::
	   :toctree: _autosummary

        .. automodule:: par_trans.manifolds.flag
	.. autofunction:: solve_w			
	.. autoclass:: Flag
	   :members:
		       
   
	:math:`\mathrm{GL}^+(n)`
	========================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: par_trans.manifolds.glp_beta
	.. autoclass:: GLpBeta
	   :members:		       

	:math:`\mathrm{SO}(n)`
	========================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: par_trans.manifolds.so_alpha
	.. autoclass:: SOAlpha
	   :members:		       


par\_trans.expv
===============

.. automodule:: par_trans.utils

	Expv
	====
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: par_trans.utils.expm_multiply_np
	   :members:

   
	Utils
	=============================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: par_trans.utils.utils
	   :members:

		       
