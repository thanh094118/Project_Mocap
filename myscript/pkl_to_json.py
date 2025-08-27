#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKL to JSON Converter
Chuyển đổi file .pkl sang .json với xử lý đầy đủ các kiểu dữ liệu
"""

import pickle
import json
import sys
import os
import argparse
from datetime import datetime, date, time
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import warnings

class AdvancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder để xử lý các kiểu dữ liệu đặc biệt"""
    
    def default(self, obj):
        # Xử lý numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.datetime64):
            return str(obj)
        
        # Xử lý scipy sparse matrices
        elif hasattr(obj, 'toarray') and hasattr(obj, 'nnz'):  # scipy sparse matrix
            return {
                '_type': f'scipy_sparse_{obj.__class__.__name__}',
                'shape': obj.shape,
                'nnz': obj.nnz,
                'data': obj.toarray().tolist(),
                'format': obj.format if hasattr(obj, 'format') else 'unknown'
            }
        
        # Xử lý pandas types
        elif isinstance(obj, pd.DataFrame):
            return {
                '_type': 'pandas_dataframe',
                'data': obj.to_dict('records'),
                'index': obj.index.tolist(),
                'columns': obj.columns.tolist()
            }
        elif isinstance(obj, pd.Series):
            return {
                '_type': 'pandas_series',
                'data': obj.to_dict(),
                'index': obj.index.tolist(),
                'name': obj.name
            }
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'to_pydatetime'):  # pandas datetime
            return obj.to_pydatetime().isoformat()
        
        # Xử lý datetime types
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()
        
        # Xử lý Decimal
        elif isinstance(obj, Decimal):
            return float(obj)
        
        # Xử lý set
        elif isinstance(obj, set):
            return {
                '_type': 'set',
                'data': list(obj)
            }
        
        # Xử lý frozenset
        elif isinstance(obj, frozenset):
            return {
                '_type': 'frozenset',
                'data': list(obj)
            }
        
        # Xử lý tuple
        elif isinstance(obj, tuple):
            return {
                '_type': 'tuple',
                'data': list(obj)
            }
        
        # Xử lý complex numbers
        elif isinstance(obj, complex):
            return {
                '_type': 'complex',
                'real': obj.real,
                'imag': obj.imag
            }
        
        # Xử lý bytes
        elif isinstance(obj, bytes):
            try:
                # Thử decode với utf-8 trước
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # Thử decode với latin-1
                    return obj.decode('latin-1')
                except UnicodeDecodeError:
                    # Nếu không decode được, chuyển thành base64
                    import base64
                    return {
                        '_type': 'bytes_base64',
                        'data': base64.b64encode(obj).decode('ascii')
                    }
        
        # Xử lý bytearray
        elif isinstance(obj, bytearray):
            return {
                '_type': 'bytearray',
                'data': list(obj)
            }
        
        # Xử lý range
        elif isinstance(obj, range):
            return {
                '_type': 'range',
                'start': obj.start,
                'stop': obj.stop,
                'step': obj.step
            }
        
        # Xử lý các object có __dict__
        elif hasattr(obj, '__dict__'):
            result = {
                '_type': f'{obj.__class__.__module__}.{obj.__class__.__name__}',
                'attributes': {}
            }
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Bỏ qua private attributes
                    try:
                        json.dumps(value, cls=AdvancedJSONEncoder)  # Test serializable
                        result['attributes'][key] = value
                    except (TypeError, ValueError):
                        result['attributes'][key] = f"<non-serializable: {type(value).__name__}>"
            return result
        
        # Xử lý callable objects
        elif callable(obj):
            return f"<function: {getattr(obj, '__name__', str(obj))}>"
        
        # Fallback cho các object khác
        else:
            return f"<{type(obj).__name__}: {str(obj)}>"

def analyze_data_structure(data, max_depth=3, current_depth=0):
    """Phân tích cấu trúc dữ liệu"""
    if current_depth >= max_depth:
        return f"<max_depth_reached: {type(data).__name__}>"
    
    analysis = {
        'type': type(data).__name__,
        'size': None,
        'sample': None
    }
    
    if isinstance(data, (list, tuple)):
        analysis['size'] = len(data)
        if data:
            analysis['sample'] = analyze_data_structure(data[0], max_depth, current_depth + 1)
    elif isinstance(data, dict):
        analysis['size'] = len(data)
        analysis['keys'] = list(data.keys())[:10]  # First 10 keys
        if data:
            first_key = next(iter(data.keys()))
            analysis['sample'] = {
                'key': first_key,
                'value': analyze_data_structure(data[first_key], max_depth, current_depth + 1)
            }
    elif isinstance(data, str):
        analysis['size'] = len(data)
        analysis['sample'] = data[:100] + ('...' if len(data) > 100 else '')
    elif hasattr(data, '__len__'):
        try:
            analysis['size'] = len(data)
        except:
            pass
    
    return analysis

def convert_pkl_to_json(input_file, output_file=None, pretty=True, analyze=True, safe_mode=True):
    """
    Chuyển đổi file .pkl sang .json
    
    Args:
        input_file (str): Đường dẫn file .pkl
        output_file (str, optional): Đường dẫn file .json output
        pretty (bool): Format JSON đẹp
        analyze (bool): Hiển thị phân tích cấu trúc
        safe_mode (bool): Chế độ an toàn (không load code tùy ý)
    """
    
    # Kiểm tra file input
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File không tồn tại: {input_file}")
    
    # Tạo tên file output nếu không được cung cấp
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.json"
    
    print(f"🔄 Đang chuyển đổi: {input_file} -> {output_file}")
    
    try:
        # Đọc file pickle
        print("📖 Đang đọc file pickle...")
        
        if safe_mode:
            # Chế độ an toàn - hạn chế các module có thể được import
            class RestrictedUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Danh sách các module được phép
                    safe_modules = [
                        'builtins', '__builtin__', 'copy_reg', 'copyreg',
                        'numpy', 'pandas', 'datetime', 'decimal', 'collections',
                        'scipy', 'scipy.sparse'
                    ]
                    if module.split('.')[0] in safe_modules:
                        return super().find_class(module, name)
                    raise pickle.UnpicklingError(f"Không được phép import module: {module}")
            
            with open(input_file, 'rb') as f:
                unpickler = RestrictedUnpickler(f)
                unpickler.encoding = 'latin-1'  # Set encoding to handle binary data
                data = unpickler.load()
        else:
            with open(input_file, 'rb') as f:
                # Thử với encoding khác nhau
                try:
                    data = pickle.load(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    data = pickle.load(f, encoding='latin-1')
                except Exception as e:
                    f.seek(0)
                    try:
                        data = pickle.load(f, encoding='bytes')
                    except:
                        raise e
        
        print("✅ Đọc file pickle thành công!")
        
        # Phân tích cấu trúc nếu được yêu cầu
        if analyze:
            print("\n📊 Phân tích cấu trúc dữ liệu:")
            structure = analyze_data_structure(data)
            print(json.dumps(structure, indent=2, cls=AdvancedJSONEncoder, ensure_ascii=False))
        
        # Chuyển đổi sang JSON
        print("\n🔄 Đang chuyển đổi sang JSON...")
        
        if pretty:
            json_str = json.dumps(
                data, 
                cls=AdvancedJSONEncoder, 
                indent=2, 
                ensure_ascii=False,
                sort_keys=True
            )
        else:
            json_str = json.dumps(
                data, 
                cls=AdvancedJSONEncoder, 
                ensure_ascii=False,
                separators=(',', ':')
            )
        
        # Ghi file JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        print(f"✅ Chuyển đổi thành công!")
        print(f"📁 File output: {output_file}")
        print(f"📏 Kích thước: {len(json_str):,} ký tự")
        
        # Hiển thị preview
        if len(json_str) > 1000:
            print(f"\n👀 Preview (1000 ký tự đầu):")
            print(json_str[:1000] + "...")
        else:
            print(f"\n👀 Nội dung:")
            print(json_str)
            
    except pickle.UnpicklingError as e:
        print(f"❌ Lỗi khi đọc pickle: {e}")
        if safe_mode:
            print("💡 Thử chạy lại với --no-safe-mode nếu bạn tin tưởng file này")
        raise
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise

def batch_convert(input_dir, output_dir=None, **kwargs):
    """Chuyển đổi hàng loạt các file .pkl trong thư mục"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {input_dir}")
    
    # Tạo thư mục output nếu không được cung cấp
    if output_dir is None:
        output_path = input_path / "json_output"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    # Tìm tất cả file .pkl
    pkl_files = list(input_path.glob("*.pkl")) + list(input_path.glob("*.pickle"))
    
    if not pkl_files:
        print("❌ Không tìm thấy file .pkl nào trong thư mục")
        return
    
    print(f"🔍 Tìm thấy {len(pkl_files)} file(s) .pkl")
    
    success_count = 0
    for pkl_file in pkl_files:
        try:
            output_file = output_path / f"{pkl_file.stem}.json"
            print(f"\n--- Xử lý: {pkl_file.name} ---")
            
            convert_pkl_to_json(str(pkl_file), str(output_file), **kwargs)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {pkl_file.name}: {e}")
    
    print(f"\n🎉 Hoàn thành! {success_count}/{len(pkl_files)} file được chuyển đổi thành công")

def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi file .pkl sang .json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python pkl_to_json.py data.pkl                    # Chuyển đổi một file
  python pkl_to_json.py data.pkl -o output.json     # Chỉ định file output
  python pkl_to_json.py -d ./data_folder             # Chuyển đổi hàng loạt
  python pkl_to_json.py data.pkl --no-pretty        # JSON không format
  python pkl_to_json.py data.pkl --no-safe-mode     # Tắt chế độ an toàn
        """
    )
    
    # Arguments chính
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("input_file", nargs="?", help="File .pkl cần chuyển đổi")
    group.add_argument("-d", "--directory", help="Thư mục chứa các file .pkl")
    
    # Options
    parser.add_argument("-o", "--output", help="File JSON output")
    parser.add_argument("--output-dir", help="Thư mục output cho batch convert")
    parser.add_argument("--no-pretty", action="store_true", help="Không format JSON đẹp")
    parser.add_argument("--no-analyze", action="store_true", help="Không phân tích cấu trúc")
    parser.add_argument("--no-safe-mode", action="store_true", help="Tắt chế độ an toàn")
    
    args = parser.parse_args()
    
    # Suppress pandas warnings
    warnings.filterwarnings('ignore')
    
    try:
        if args.directory:
            # Batch convert
            batch_convert(
                args.directory,
                args.output_dir,
                pretty=not args.no_pretty,
                analyze=not args.no_analyze,
                safe_mode=not args.no_safe_mode
            )
        else:
            # Single file convert
            convert_pkl_to_json(
                args.input_file,
                args.output,
                pretty=not args.no_pretty,
                analyze=not args.no_analyze,
                safe_mode=not args.no_safe_mode
            )
            
    except KeyboardInterrupt:
        print("\n❌ Đã hủy bởi người dùng")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()