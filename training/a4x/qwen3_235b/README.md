# To run this recipe:

1. SSH into the Slurm login node
2. Copy this recipe folder to `/home/$USER/recipe`

**Note:** You must replace `<YOUR_HF_TOKEN>` in `launch_script.sh` with your own Hugging Face token before running this recipe.

3. From your home directory (`/home/$USER`), run:
   ```
   sbatch recipe/sbatch_script.sh
   ```
