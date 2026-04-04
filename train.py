import datetime
import traceback

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.engine.suite import ForgeSuite


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    """
    Orchestrates MARL experiments through the ForgeSuite engine.
    """
    run_list = cfg.get('algorithms', [cfg.algorithm])
    if isinstance(run_list, str):
        import ast

        run_list = ast.literal_eval(run_list) if '[' in run_list else [run_list]

    l_list = (
        OmegaConf.to_container(run_list)
        if not isinstance(run_list, (list, str))
        else run_list
    )
    is_competitive = (
        len(l_list) == 2 and isinstance(l_list[0], list) and isinstance(l_list[1], list)
    )

    pairings = (
        list(zip(l_list[0], l_list[1], strict=False))
        if is_competitive
        else [(None, alg) for alg in l_list]
    )
    print(f'[NeuralForge] Initializing Pairings: {pairings}')

    for red_alg, blue_alg in pairings:
        alg_label = f'{red_alg}_vs_{blue_alg}' if is_competitive else blue_alg
        run_cfg = OmegaConf.merge(cfg)
        run_cfg.blue_algorithm = blue_alg
        run_cfg.red_algorithm = red_alg

        start_time = datetime.datetime.now().strftime('%H-%M')
        run_cfg.timestamp = f'{alg_label.upper()}_{start_time}'

        print(f'\n{"=" * 60}')
        print(f'Launching: {run_cfg.timestamp}')
        print(f'{"=" * 60}\n')

        suite = ForgeSuite(run_cfg)
        try:
            suite.train()
        except Exception:
            print(f'[NeuralForge] System Failure: {run_cfg.timestamp}')
            traceback.print_exc()
        finally:
            suite.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
