# DMC transfer pilot implementation plan

1. Add strict DMC configuration and experiment-matrix types that encode the two
   approved transfer chains, three methods, and equal charged budgets.
2. Add an optional `dm-control` dependency and a Gymnasium adapter with stable
   observation flattening, seeding, action repeat, and episode termination.
3. Add a DMC source episode store and ETL bundle builder with independent ETL
   latent and SAR rank settings.
4. Add matched SAC factories for direct Lattice and latent ETL target training,
   frozen evaluation normalization, periodic evaluation, checkpoint/resume,
   and return-curve aggregation.
5. Add a sequential server launcher and two YAML configs under an isolated
   output root.
6. Fix MuJoCo model/data resolution for MyoSuite wrappers without changing the
   metric definition.
7. Run focused tests, the complete non-MyoSuite suite, shell syntax checks, and
   line-ending checks; then update README/server instructions and commit.
