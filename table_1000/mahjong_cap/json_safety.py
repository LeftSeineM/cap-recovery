"""Convert numpy/tensor scalars, including bool_, without changing numeric values."""
def plain(value):
 if value is None or isinstance(value,(str,bool,int,float)):return value
 if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [plain(v) for v in value]
 if hasattr(value,'detach'):value=value.detach().cpu().numpy()
 if hasattr(value,'tolist'):return plain(value.tolist())
 if hasattr(value,'item'):return plain(value.item())
 return {'unserialized_type':type(value).__name__,'repr':repr(value)}
