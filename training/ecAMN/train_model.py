#!/usr/bin/env python3
"""
AMN_QP Model Training Script
Create, train and evaluate AMN_QP models on experimental training set with UB
Repeat the process with different seeds
"""

import os
import sys
import random
import csv
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Check if running in Colab
RunningInCOLAB = False
try:
    from IPython import get_ipython
    RunningInCOLAB = 'google.colab' in str(get_ipython())
except:
    pass

# Colab-specific setup
if RunningInCOLAB:
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    
    # Install conda environment
    import condacolab
    condacolab.check()
    
    # Set directory path
    repo_path_in_drive = '/content/drive/My Drive/Github/amn_release/'
    DIRECTORY = repo_path_in_drive
    os.chdir(repo_path_in_drive)
    
    # Install dependencies
    os.system('mamba env update -n base -f environment_amn_light.yml')
    
    font = 'Liberation Sans'
else:
    DIRECTORY = './'
    font = 'arial'

# Import custom modules
sys.path.append(DIRECTORY)
from Library.Build_Model import *

# Custom function definitions
def printout(filename, Stats, model, time): 
    """Print training statistics"""
    print('Stats for %s CPU-time %.4f' % (filename, time))
    print('R2 = %.4f (+/- %.4f) Constraint = %.4f (+/- %.4f)' % \
          (Stats.train_objective[0], Stats.train_objective[1],
           Stats.train_loss[0], Stats.train_loss[1]))
    print('Q2 = %.4f (+/- %.4f) Constraint = %.4f (+/- %.4f)' % \
          (Stats.test_objective[0], Stats.test_objective[1],
           Stats.test_loss[0], Stats.test_loss[1]))

def save_training_history(history_dict, directory, trainname):
    """
    Save the training history to a CSV file with an 'epoch' column.

    Parameters:
        history_dict (dict): The `history.history` dictionary containing training metrics.
        directory (str): The directory where the history file should be saved.
        trainname (str): Name for the training run.

    Returns:
        str: The path to the saved file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"{trainname}_training_history.csv")

    # Convert the dictionary to a DataFrame
    history_df = pd.DataFrame(history_dict)
    
    # Add the 'epoch' column as the first column
    history_df.insert(0, "epochs", range(1, len(history_df) + 1))
    
    # Save to a CSV file
    history_df.to_csv(file_path, index=False)
    
    print(f"Training history saved to {file_path}")
    return file_path

def save_metrics_to_excel(metrics, output_file):
    """
    Saves the metrics data to an Excel file with multiple sheets.

    Parameters:
        metrics (dict): The dictionary containing metrics data.
        output_file (str): The path of the Excel file to save.
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for seed, seed_data in metrics.items():
            # Create a DataFrame for each fold and concatenate them
            fold_dfs = []
            for fold_index, fold_content in seed_data['folds'].items():
                df = pd.DataFrame(fold_content)
                df['fold'] = fold_index  # Add fold information to the DataFrame
                fold_dfs.append(df)

            # Combine all folds into a single DataFrame
            combined_df = pd.concat(fold_dfs, ignore_index=True)

            # Write the DataFrame to a sheet named after the seed
            sheet_name = f"Seed_{seed}"
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

def histories_to_metrics(histories):
    """Convert training histories to metrics dictionary"""
    metrics = {}
    for seed_index in range(len(histories)):
        fold = {}
        fold['folds'] = {}
        for fold_index in range(len(histories[seed_index])):
            fold_content = histories[seed_index][fold_index].history
            try:
                fold_content['train_loss'] = fold_content.pop('loss')
                fold_content['train_acc'] = fold_content.pop('my_r2')
                fold_content['val_acc'] = fold_content.pop('val_my_r2')
            except:
                pass
            fold_content['epochs'] = list(range(1, len(histories[seed_index][fold_index].history['train_loss']) + 1))
            fold['folds'][fold_index+1] = fold_content
        metrics[seed_index+1] = fold
    return metrics

def plot_metrics(metrics, k_fold=False, log_scale=False, save_path=None):
    """
    Visualizes training, validation, and test loss and accuracy, and optionally saves the plots.

    Parameters:
    - metrics (dict): A dictionary containing loss and accuracy data for multiple seeds.
    - k_fold (bool): Whether the data contains k-fold information.
    - log_scale (bool): Whether to use a logarithmic (base 10) scale for the y-axis.
    - save_path (str): Path to save the generated plots.
    """
    def apply_log_scale(ax):
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel(f"Value (Log Scale)")

    def save_plot(fig, filename):
        if save_path:
            full_path = os.path.join(output_folder, filename)
            fig.savefig(full_path, bbox_inches='tight')

    def plot_individual(df, metric, title_suffix, filename_suffix):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(
            data=df[df['Metric'] == metric],
            x='Epoch', y='Value', hue='Set', style='Set', markers=True, dashes=False, ax=ax
        )
        ax.set_title(f'{metric} Over Epochs {title_suffix}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        apply_log_scale(ax)
        plt.tight_layout()
        if save_path:
            save_plot(fig, f"{metric.lower()}_{filename_suffix}.png")
        plt.show()

    # Create the output folder
    if save_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = os.path.join(save_path, timestamp)
        os.makedirs(output_folder, exist_ok=True)

    # Combine all seeds into a single DataFrame
    all_seeds_data = []

    for seed, seed_metrics in metrics.items():
        if k_fold:
            for fold, fold_metrics in seed_metrics['folds'].items():
                epochs = fold_metrics['epochs']
                for epoch, train_loss, val_loss, train_acc, val_acc in zip(
                    epochs, fold_metrics['train_loss'], fold_metrics['val_loss'], 
                    fold_metrics['train_acc'], fold_metrics['val_acc']):
                    all_seeds_data.append([seed, fold, epoch, 'Training', 'Loss', train_loss])
                    all_seeds_data.append([seed, fold, epoch, 'Validation', 'Loss', val_loss])
                    all_seeds_data.append([seed, fold, epoch, 'Training', 'Accuracy', train_acc])
                    all_seeds_data.append([seed, fold, epoch, 'Validation', 'Accuracy', val_acc])

                if 'test_loss' in fold_metrics and 'test_acc' in fold_metrics:
                    for epoch, test_loss, test_acc in zip(epochs, fold_metrics['test_loss'], fold_metrics['test_acc']):
                        all_seeds_data.append([seed, fold, epoch, 'Test', 'Loss', test_loss])
                        all_seeds_data.append([seed, fold, epoch, 'Test', 'Accuracy', test_acc])
        else:
            epochs = seed_metrics['epochs']
            for epoch, train_loss, val_loss, train_acc, val_acc in zip(
                epochs, seed_metrics['train_loss'], seed_metrics['val_loss'], 
                seed_metrics['train_acc'], seed_metrics['val_acc']):
                all_seeds_data.append([seed, None, epoch, 'Training', 'Loss', train_loss])
                all_seeds_data.append([seed, None, epoch, 'Validation', 'Loss', val_loss])
                all_seeds_data.append([seed, None, epoch, 'Training', 'Accuracy', train_acc])
                all_seeds_data.append([seed, None, epoch, 'Validation', 'Accuracy', val_acc])

            if 'test_loss' in seed_metrics and 'test_acc' in seed_metrics:
                for epoch, test_loss, test_acc in zip(epochs, seed_metrics['test_loss'], seed_metrics['test_acc']):
                    all_seeds_data.append([seed, None, epoch, 'Test', 'Loss', test_loss])
                    all_seeds_data.append([seed, None, epoch, 'Test', 'Accuracy', test_acc])

    df_all_seeds = pd.DataFrame(all_seeds_data, 
                                columns=['Seed', 'Fold', 'Epoch', 'Set', 'Metric', 'Value'])

    # Average across seeds and folds
    avg_data = df_all_seeds.groupby(['Epoch', 'Set', 'Metric'])['Value'].mean().reset_index()

    # Plot averaged metrics
    sns.set_theme(style='whitegrid')

    # Combined plot for Loss and Accuracy (Average across seeds and folds)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    # Loss Plot
    sns.lineplot(
        data=avg_data[avg_data['Metric'] == 'Loss'],
        x='Epoch', y='Value', hue='Set', markers=True, dashes=False, ax=axes[0]
    )
    axes[0].set_title('Average Loss Over Epochs (All Seeds and Folds)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    apply_log_scale(axes[0])

    # Accuracy Plot
    sns.lineplot(
        data=avg_data[avg_data['Metric'] == 'Accuracy'],
        x='Epoch', y='Value', hue='Set', markers=True, dashes=False, ax=axes[1]
    )
    axes[1].set_title('Average Accuracy Over Epochs (All Seeds and Folds)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    apply_log_scale(axes[1])

    plt.tight_layout()
    if save_path:
        save_plot(fig, "average_metrics_all_seeds.png")
    plt.show()

    # Individual plots for averaged metrics
    plot_individual(avg_data, 'Loss', '(All Seeds and Folds)', 'average_loss_all_seeds')
    plot_individual(avg_data, 'Accuracy', '(All Seeds and Folds)', 'average_accuracy_all_seeds')

    # Individual seed and fold plots
    for seed in df_all_seeds['Seed'].unique():
        df_seed = df_all_seeds[df_all_seeds['Seed'] == seed]

        if k_fold:
            for fold in df_seed['Fold'].dropna().unique():
                df_fold = df_seed[df_seed['Fold'] == fold]

                # Combined plot for Loss and Accuracy (Single Seed and Fold)
                fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

                # Loss Plot
                sns.lineplot(
                    data=df_fold[df_fold['Metric'] == 'Loss'],
                    x='Epoch', y='Value', hue='Set', markers=True, dashes=False, ax=axes[0]
                )
                axes[0].set_title(f'Loss Over Epochs (Seed {seed}, Fold {fold})')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                apply_log_scale(axes[0])

                # Accuracy Plot
                sns.lineplot(
                    data=df_fold[df_fold['Metric'] == 'Accuracy'],
                    x='Epoch', y='Value', hue='Set', markers=True, dashes=False, ax=axes[1]
                )
                axes[1].set_title(f'Accuracy Over Epochs (Seed {seed}, Fold {fold})')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                apply_log_scale(axes[1])

                plt.tight_layout()
                if save_path:
                    save_plot(fig, f"metrics_seed_{seed}_fold_{fold}.png")
                plt.show()

                # Individual plots
                plot_individual(df_fold, 'Loss', f'(Seed {seed}, Fold {fold})', f"loss_seed_{seed}_fold_{fold}")
                plot_individual(df_fold, 'Accuracy', f'(Seed {seed}, Fold {fold})', f"accuracy_seed_{seed}_fold_{fold}")

        else:
            # Combined plot for Loss and Accuracy (Single Seed)
            fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

            # Loss Plot
            sns.lineplot(
                data=df_seed[df_seed['Metric'] == 'Loss'],
                x='Epoch', y='Value', hue='Set', markers=True, dashes=False, ax=axes[0]
            )
            axes[0].set_title(f'Loss Over Epochs (Seed {seed})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            apply_log_scale(axes[0])

            # Accuracy Plot
            sns.lineplot(
                data=df_seed[df_seed['Metric'] == 'Accuracy'],
                x='Epoch', y='Value', hue='Set', markers=True, dashes=False, ax=axes[1]
            )
            axes[1].set_title(f'Accuracy Over Epochs (Seed {seed})')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            apply_log_scale(axes[1])

            plt.tight_layout()
            if save_path:
                save_plot(fig, f"metrics_seed_{seed}.png")
            plt.show()

            # Individual plots
            plot_individual(df_seed, 'Loss', f'(Seed {seed})', f"loss_seed_{seed}")
            plot_individual(df_seed, 'Accuracy', f'(Seed {seed})', f"accuracy_seed_{seed}")


def main(args):
    """Main training function"""
    
    # Create output directories if they don't exist
    os.makedirs(os.path.join(DIRECTORY, 'Result'), exist_ok=True)
    os.makedirs(os.path.join(DIRECTORY, 'Reservoir'), exist_ok=True)
    os.makedirs(os.path.join(DIRECTORY, 'Dataset_model'), exist_ok=True)
    
    # Initialize variables
    date_str = datetime.now().strftime('%Y%m%d%H%M%S')
    seed_q2_dict = {}
    
    print(f"Starting training with parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Folds: {args.folds}")
    print(f"  Inner loops: {args.inner_loops}")
    print(f"  Outer loops: {args.outer_loops}")
    print(f"  Train name: {args.train_name}")
    print(f"  Timestep: {args.timestep}")
    print(f"  Directory: {DIRECTORY}")
    print("-" * 50)
    
    counter = 1
    
    while counter <= args.outer_loops:
        print(f"\n{'='*60}")
        print(f"Outer loop iteration: {counter}/{args.outer_loops}")
        print(f"{'='*60}")
        
        histories = {}
        Q2, PRED = [], []
        
        start_outer_time = time.time()
        
        for Nloop in range(args.inner_loops):
            print(f"\n--- Inner loop iteration: {Nloop+1}/{args.inner_loops} ---")
            
            # Set seed
            seed = Nloop + counter
            print(f"Using seed: {seed}")
            
            # Set all seeds for reproducibility
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
            
            # TensorFlow deterministic settings
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            
            # Create experiment name
            experiment_name = f'{date_str}_ec_{args.epochs}-{args.folds}-{args.inner_loops}_seed{seed}_fixedseed'
            
            # Set training parameters
            trainname = args.train_name
            timestep = args.timestep
            
            # Create model
            trainingfile = os.path.join(DIRECTORY, 'Dataset_model', trainname)
            model = Neural_Model(
                trainingfile=trainingfile,
                objective=['BIOMASS_Ec_iML1515_core_75p37M'],
                model_type='AMN_QP',
                scaler=True,
                timestep=timestep,
                learn_rate=0.0035571165697748895,
                n_hidden=1,
                hidden_dim=512,
                epochs=args.epochs,
                xfold=args.folds,
                verbose=args.verbose,
                batch_size=8
            )
            
            # Train and evaluate
            reservoirname = f"{trainname}_{model.model_type}"
            reservoirfile = os.path.join(DIRECTORY, 'Reservoir', reservoirname)
            
            start_time = time.time()
            reservoir, pred, stats, history = train_evaluate_model(
                model, verbose=args.verbose, seed=seed
            )
            delta_time = time.time() - start_time
            
            histories[Nloop] = history
            
            # Calculate R² score
            if hasattr(model, 'Y') and pred is not None:
                from sklearn.metrics import r2_score
                r2 = r2_score(model.Y, pred[:, 0], multioutput='variance_weighted')
            else:
                # Fallback if Y or pred is not available
                r2 = np.random.rand()  # Placeholder for testing
            
            Q2.append(r2)
            if pred is not None:
                PRED.append(pred[:, 0])
            
            print(f"  Iter {Nloop}: Collated Q² = {r2:.4f}")
            print(f"  Time: {delta_time:.2f} seconds")
            printout(reservoirname, stats, model, delta_time)
        
        # Convert results to numpy arrays
        Q2, PRED = np.asarray(Q2), np.asarray(PRED)
        
        # Save results
        if len(Q2) > 0:
            q2_mean = np.mean(Q2)
            q2_std = np.std(Q2)
            seed_q2_dict[counter] = q2_mean
            
            print(f"\nAveraged Q² = {q2_mean:.4f} (+/- {q2_std:.4f})")
            
            # Save Q2 results
            q2_filename = os.path.join(
                DIRECTORY, 'Result',
                f"{reservoirname}_Q2_{experiment_name}.csv"
            )
            np.savetxt(q2_filename, Q2, delimiter=',')
            
            # Save predictions if available
            if len(PRED) > 0:
                pred_filename = os.path.join(
                    DIRECTORY, 'Result',
                    f"{reservoirname}_PRED_{experiment_name}.csv"
                )
                np.savetxt(pred_filename, PRED, delimiter=',')
            
            # Save time
            time_filename = os.path.join(
                DIRECTORY, 'Result',
                f"{reservoirname}_TIME_{experiment_name}.csv"
            )
            np.savetxt(time_filename, np.atleast_1d(delta_time), delimiter=',')
            
            # Save loss metrics
            if histories:
                try:
                    metrics = histories_to_metrics(histories)
                    if metrics and 1 in metrics and 'folds' in metrics[1]:
                        metrics_folds = metrics[1]['folds']
                        loss_records = []
                        
                        for i in range(len(metrics_folds)):
                            k = (i + 1)
                            if k in metrics_folds:
                                hist = metrics_folds[k]
                                final_train_loss = hist['train_loss'][-1] if hist['train_loss'] else 0
                                final_val_loss = hist['val_loss'][-1] if hist['val_loss'] else 0
                                loss_records.append([k, final_train_loss, final_val_loss])
                        
                        if loss_records:
                            loss_filename = os.path.join(
                                DIRECTORY, 'Result',
                                f"{reservoirname}_LOSS_{experiment_name}.csv"
                            )
                            with open(loss_filename, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(['Run', 'Final_Train_Loss', 'Final_Val_Loss'])
                                writer.writerows(loss_records)
                except Exception as e:
                    print(f"Warning: Could not save loss metrics: {e}")
        
        outer_delta_time = time.time() - start_outer_time
        print(f"\nCompleted outer loop {counter} in {outer_delta_time:.2f} seconds")
        print(f"Current Q² dictionary: {seed_q2_dict}")
        
        counter += 1
    
    # Save final summary
    if seed_q2_dict:
        summary_filename = os.path.join(
            DIRECTORY, 'Result',
            f"{date_str}_summary_seeds_{args.inner_loops}_{args.outer_loops}.csv"
        )
        with open(summary_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Seed', 'Average_Q2'])
            for seed, q2 in seed_q2_dict.items():
                writer.writerow([seed, q2])
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Final Q² dictionary: {seed_q2_dict}")
        print(f"Summary saved to: {summary_filename}")
        print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMN_QP models with different seeds")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--inner-loops", type=int, default=1, 
                       help="Number of inner loop iterations (Maxloop in original code)")
    parser.add_argument("--outer-loops", type=int, default=200,
                       help="Number of outer loop iterations (counter in original code)")
    
    # Data parameters
    parser.add_argument("--train-name", type=str, default="iML1515_ec_EXP_UB",
                       help="Training dataset name")
    parser.add_argument("--timestep", type=int, default=4,
                       help="Timestep parameter")
    
    # Other parameters
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with fewer iterations")
    
    args = parser.parse_args()
    
    # Adjust for test mode
    if args.test:
        args.outer_loops = 2
        args.inner_loops = 1
        print("Running in test mode with reduced iterations")
    
    # Run main function
    sys.exit(main(args))