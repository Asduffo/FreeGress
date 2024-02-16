NAMES=${1//[^[:alnum:]]/}


for i in 100 200 300 400 500 600 700 800 900 1000
do
    python3 guidance/main_guidance.py +experiment=zinc250k.yaml dataset=zinc250k general.wandb="online" general.name="cb_zinc_"$NAMES"_test_"$i \
    guidance.guidance_target=\[$1\] +general.test_samples_to_generate=100 \
    +general.test_only=$2 general.gpus=[1] guidance.guidance_medium="NONE" train.batch_size=1  +general.trained_regressor_path=$3 \
    +guidance.lambda_guidance=$i; 
done
