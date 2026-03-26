# To run this recipe:

**Note**: Before running this recipe, please open `launch_script.sh` and replace `<YOUR_HUGGINGFACE_TOKEN>` with your actual HuggingFace token.

1. SSH into the Slurm login node
2. Copy this recipe folder to `/home/$USER/recipe`
3. From your home directory (`/home/$USER`), run:
   ```
   sbatch recipe/sbatch_script.sh
   ```
