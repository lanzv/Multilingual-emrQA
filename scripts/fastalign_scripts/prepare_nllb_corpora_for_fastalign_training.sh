#!/bin/sh
#SBATCH -J nllb
#SBATCH -o scripts/slurm_outputs/nllb.out
#SBATCH -p cpu-ms


sbatch --job-name=bg_nllb \
     --output=scripts/slurm_outputs/nllb/bg_nllb.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu1 <<"EOF"
#!/bin/bash

python3 - <<END

src_nllb, src_language, src_code = "bul_Cyrl", "bulgarian", "bg"
tgt_nllb, tgt_language, tgt_code = "eng_Latn", "english", "en"

from datasets import load_dataset
from src.utils import tokenize
dataset = load_dataset("allenai/nllb", "{}-{}".format(src_nllb, tgt_nllb), streaming=True)
with open("../datasets/nllb/{}{}_nllb.txt".format(src_code, tgt_code), "a") as f:
    counter = 0
    for example in dataset['train']:
        if counter == 44635282:
            break
        counter += 1
        src = ' '.join(tokenize(example["translation"][src_nllb], language=src_language, warnings = False)[0])
        tgt = ' '.join(tokenize(example["translation"][tgt_nllb], language=tgt_language, warnings = False)[0])
        dataset_line = "{} ||| {}".format(src, tgt)
        f.write(dataset_line + "\n")
    print("{}{}_nllb sentences: {}".format(src_code, tgt_code, counter))
END
EOF


sbatch --job-name=cs_nllb \
     --output=scripts/slurm_outputs/nllb/cs_nllb.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu2 <<"EOF"
#!/bin/bash

python3 - <<END

src_nllb, src_language, src_code = "ces_Latn", "czech", "cs"
tgt_nllb, tgt_language, tgt_code = "eng_Latn", "english", "en"

from datasets import load_dataset
from src.utils import tokenize
import nltk
nltk.download('punkt')
dataset = load_dataset("allenai/nllb", "{}-{}".format(src_nllb, tgt_nllb), streaming=True)
with open("../datasets/nllb/{}{}_nllb.txt".format(src_code, tgt_code), "a") as f:
    counter = 0
    for example in dataset['train']:
        if counter == 44635282:
            break
        counter += 1
        src = ' '.join(tokenize(example["translation"][src_nllb], language=src_language, warnings = False)[0])
        tgt = ' '.join(tokenize(example["translation"][tgt_nllb], language=tgt_language, warnings = False)[0])
        dataset_line = "{} ||| {}".format(src, tgt)
        f.write(dataset_line + "\n")
    print("{}{}_nllb sentences: {}".format(src_code, tgt_code, counter))
END
EOF


sbatch --job-name=el_nllb \
     --output=scripts/slurm_outputs/nllb/el_nllb.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash

python3 - <<END

src_nllb, src_language, src_code = "ell_Grek", "greek", "el"
tgt_nllb, tgt_language, tgt_code = "eng_Latn", "english", "en"

from datasets import load_dataset
from src.utils import tokenize
dataset = load_dataset("allenai/nllb", "{}-{}".format(src_nllb, tgt_nllb), streaming=True)
with open("../datasets/nllb/{}{}_nllb.txt".format(src_code, tgt_code), "a") as f:
    counter = 0
    for example in dataset['train']:
        if counter == 44635282:
            break
        counter += 1
        src = ' '.join(tokenize(example["translation"][src_nllb], language=src_language, warnings = False)[0])
        tgt = ' '.join(tokenize(example["translation"][tgt_nllb], language=tgt_language, warnings = False)[0])
        dataset_line = "{} ||| {}".format(src, tgt)
        f.write(dataset_line + "\n")
    print("{}{}_nllb sentences: {}".format(src_code, tgt_code, counter))
END
EOF



sbatch --job-name=es_nllb \
     --output=scripts/slurm_outputs/nllb/es_nllb.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-4gpu3 <<"EOF"
#!/bin/bash

python3 - <<END

tgt_nllb, tgt_language, tgt_code = "spa_Latn", "spanish", "es"
src_nllb, src_language, src_code = "eng_Latn", "english", "en"

from datasets import load_dataset
from src.utils import tokenize
dataset = load_dataset("allenai/nllb", "{}-{}".format(src_nllb, tgt_nllb), streaming=True)
with open("../datasets/nllb/{}{}_nllb.txt".format(tgt_code, src_code), "a") as f:
    counter = 0
    for example in dataset['train']:
        if counter == 44635282:
            break
        counter += 1
        src = ' '.join(tokenize(example["translation"][src_nllb], language=src_language, warnings = False)[0])
        tgt = ' '.join(tokenize(example["translation"][tgt_nllb], language=tgt_language, warnings = False)[0])
        dataset_line = "{} ||| {}".format(tgt, src)
        f.write(dataset_line + "\n")
    print("{}{}_nllb sentences: {}".format(tgt_code, src_code, counter))
END
EOF



sbatch --job-name=pl_nllb \
     --output=scripts/slurm_outputs/nllb/pl_nllb.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu5 <<"EOF"
#!/bin/bash

python3 - <<END


tgt_nllb, tgt_language, tgt_code = "pol_Latn", "polish", "pl"
src_nllb, src_language, src_code = "eng_Latn", "english", "en"

from datasets import load_dataset
from src.utils import tokenize
dataset = load_dataset("allenai/nllb", "{}-{}".format(src_nllb, tgt_nllb), streaming=True)
with open("../datasets/nllb/{}{}_nllb.txt".format(tgt_code, src_code), "a") as f:
    counter = 0
    for example in dataset['train']:
        if counter == 44635282:
            break
        counter += 1
        src = ' '.join(tokenize(example["translation"][src_nllb], language=src_language, warnings = False)[0])
        tgt = ' '.join(tokenize(example["translation"][tgt_nllb], language=tgt_language, warnings = False)[0])
        dataset_line = "{} ||| {}".format(tgt, src)
        f.write(dataset_line + "\n")
    print("{}{}_nllb sentences: {}".format(tgt_code, src_code, counter))
END
EOF




sbatch --job-name=ro_nllb \
     --output=scripts/slurm_outputs/nllb/ro_nllb.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-4gpu4 <<"EOF"
#!/bin/bash

python3 - <<END

tgt_nllb, tgt_language, tgt_code = "ron_Latn", "romanian", "ro"
src_nllb, src_language, src_code = "eng_Latn", "english", "en"

from datasets import load_dataset
from src.utils import tokenize
dataset = load_dataset("allenai/nllb", "{}-{}".format(src_nllb, tgt_nllb), streaming=True)
with open("../datasets/nllb/{}{}_nllb.txt".format(tgt_code, src_code), "a") as f:
    counter = 0
    for example in dataset['train']:
        if counter == 44635282:
            break
        counter += 1
        src = ' '.join(tokenize(example["translation"][src_nllb], language=src_language, warnings = False)[0])
        tgt = ' '.join(tokenize(example["translation"][tgt_nllb], language=tgt_language, warnings = False)[0])
        dataset_line = "{} ||| {}".format(tgt, src)
        f.write(dataset_line + "\n")
    print("{}{}_nllb sentences: {}".format(tgt_code, src_code, counter))
END
EOF