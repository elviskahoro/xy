Data-bound charts whose plan callable (e.g. a hexbin ``reduce_C_function``)
has a ``set``/``frozenset`` default no longer come back blank across workers.
The plan digest content-addresses the callable's bytecode and defaults, but
``_code_fingerprint`` hashed ``__defaults__``/``__kwdefaults__`` (and tuple/dict
bytecode constants nesting sets) through raw ``repr(...)``, whose iteration
order follows the process's ``PYTHONHASHSEED``. The frontend compile worker
then baked one digest into the page while backend-only workers registered the
same plan under a different one, every subscription missed the process-local
plan map, and the client burned its resync budget retrying the dead digest —
a permanently empty chart with only a console warning. Defaults and container
constants are now canonicalized (set/frozenset element reprs sorted, tuples
and dicts recursed) before hashing, so identical code yields one digest in
every worker regardless of the hash seed, as the fingerprint's contract always
claimed.
