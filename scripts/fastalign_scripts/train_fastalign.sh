#!/bin/sh
#SBATCH -J train_fastalign
#SBATCH -o scripts/slurm_outputs/train_fastalign.out
#SBATCH -p cpu-ms


sbatch --job-name=bg_fa_train \
     --output=scripts/slurm_outputs/nllb/bg_fa_train.out \
     --partition=cpu-ms \
     --mem-per-cpu=50G <<"EOF"
#!/bin/bash
cd ../models/fast_align/build/
./fast_align -i ../../../datasets/nllb/bgen_nllb.txt -d -v -o -p bgen/fwd_params >bgen/fwd_align 2>bgen/fwd_err
./fast_align -i ../../../datasets/nllb/bgen_nllb.txt -r -d -v -o -p bgen/rev_params >bgen/rev_align 2>bgen/rev_err
EOF


sbatch --job-name=cs_fa_train \
     --output=scripts/slurm_outputs/nllb/cs_fa_train.out \
     --partition=cpu-ms \
     --mem-per-cpu=90G <<"EOF"
#!/bin/bash

cd ../models/fast_align/build/
./fast_align -i ../../../datasets/nllb/csen_nllb.txt -d -v -o -p csen/fwd_params >csen/fwd_align 2>csen/fwd_err
./fast_align -i ../../../datasets/nllb/csen_nllb.txt -r -d -v -o -p csen/rev_params >csen/rev_align 2>csen/rev_err
EOF


sbatch --job-name=el_fa_train \
     --output=scripts/slurm_outputs/nllb/el_fa_train.out \
     --partition=cpu-ms \
     --mem-per-cpu=90G <<"EOF"
#!/bin/bash
cd ../models/fast_align/build/
./fast_align -i ../../../datasets/nllb/elen_nllb.txt -d -v -o -p elen/fwd_params >elen/fwd_align 2>elen/fwd_err
./fast_align -i ../../../datasets/nllb/elen_nllb.txt -r -d -v -o -p elen/rev_params >elen/rev_align 2>elen/rev_err
EOF



sbatch --job-name=es_fa_train \
     --output=scripts/slurm_outputs/nllb/es_fa_train.out \
     --partition=cpu-ms \
     --mem-per-cpu=90G <<"EOF"
#!/bin/bash

cd ../models/fast_align/build/
./fast_align -i ../../../datasets/nllb/esen_nllb.txt -d -v -o -p esen/fwd_params >esen/fwd_align 2>esen/fwd_err
./fast_align -i ../../../datasets/nllb/esen_nllb.txt -r -d -v -o -p esen/rev_params >esen/rev_align 2>esen/rev_err
EOF



sbatch --job-name=pl_fa_train \
     --output=scripts/slurm_outputs/nllb/pl_fa_train.out \
     --partition=cpu-ms \
     --mem-per-cpu=90G <<"EOF"
#!/bin/bash

cd ../models/fast_align/build/
./fast_align -i ../../../datasets/nllb/plen_nllb.txt -d -v -o -p plen/fwd_params >plen/fwd_align 2>plen/fwd_err
./fast_align -i ../../../datasets/nllb/plen_nllb.txt -r -d -v -o -p plen/rev_params >plen/rev_align 2>plen/rev_err
EOF




sbatch --job-name=ro_fa_train \
     --output=scripts/slurm_outputs/nllb/ro_fa_train.out \
     --partition=cpu-ms \
     --mem-per-cpu=90G <<"EOF"
#!/bin/bash

cd ../models/fast_align/build/
./fast_align -i ../../../datasets/nllb/roen_nllb.txt -d -v -o -p roen/fwd_params >roen/fwd_align 2>roen/fwd_err
./fast_align -i ../../../datasets/nllb/roen_nllb.txt -r -d -v -o -p roen/rev_params >roen/rev_align 2>roen/rev_err
EOF