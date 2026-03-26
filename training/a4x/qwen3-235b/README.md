# To run this recipe:

1. Insert your Hugging Face token into `launch_script.sh` by replacing `<YOUR_HF_TOKEN>`.
2. SSH into the Slurm login node
3. Copy this recipe folder to `/home/$USER/recipe`
4. From your home directory (`/home/$USER`), run:
   ```
   sbatch recipe/sbatch_script.sh
   ```
