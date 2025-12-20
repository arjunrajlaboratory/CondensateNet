"""Command-line interface for CondensateNet."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='condensatenet',
        description='CondensateNet: Deep learning for biomolecular condensate segmentation'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser(
        'download',
        help='Download model from HuggingFace'
    )
    download_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: ~/.cache/huggingface/condensatenet)'
    )
    download_parser.add_argument(
        '--repo-id',
        type=str,
        default='rajlab/condensatenet',
        help='HuggingFace repository ID'
    )
    download_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Re-download even if files exist'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show model information'
    )
    info_parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Model source (HuggingFace repo or local path)'
    )
    
    # Parse args
    args = parser.parse_args()
    
    if args.command == 'download':
        cmd_download(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_download(args):
    """Handle download command."""
    from .model import download_model
    
    path = download_model(
        output_dir=args.output,
        repo_id=args.repo_id,
        force=args.force
    )
    
    print(f"\nModel files saved to: {path}")
    print("\nTo use in Python:")
    print(f'  from condensatenet import CondensateNetPipeline')
    print(f'  pipeline = CondensateNetPipeline.from_local("{path}")')


def cmd_info(args):
    """Handle info command."""
    from .model import get_model_info
    from .utils import get_cache_dir
    import os
    
    print("CondensateNet Model Information")
    print("=" * 40)
    
    try:
        info = get_model_info(args.source)
        print(f"Repository:         {info['repo_id']}")
        print(f"Model type:         {info['model_type']}")
        print(f"Encoder variant:    {info['encoder_variant']}")
        print(f"Pyramid channels:   {info['pyramid_channels']}")
        print(f"Spatial attention:  {info['use_spatial_attention']}")
        print(f"Dropout rate:       {info['dropout_rate']}")
    except Exception as e:
        print(f"Error loading model info: {e}")
    
    print()
    print("Cache Information")
    print("-" * 40)
    cache_dir = get_cache_dir()
    print(f"Cache directory:    {cache_dir}")
    print(f"Cache exists:       {os.path.isdir(cache_dir)}")
    
    if os.path.isdir(cache_dir):
        files = os.listdir(cache_dir)
        print(f"Cached files:       {', '.join(files) if files else '(empty)'}")


if __name__ == '__main__':
    main()
