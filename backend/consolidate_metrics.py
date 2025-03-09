import json
import statistics

def collect_metrics(collector, metrics):
    for key, value in metrics.items():
        if isinstance(value, dict):
            if key not in collector:
                collector[key] = {}
            collect_metrics(collector[key], value)
        else:
            if key not in collector:
                collector[key] = []
            collector[key].append(float(value))

def process_collector(current_collector):
    result = {}
    for key, value in current_collector.items():
        if isinstance(value, dict):
            result[key] = process_collector(value)
        else:
            mean = statistics.mean(value) if value else 0.0
            try:
                var = statistics.pvariance(value) if value else 0.0
            except statistics.StatisticsError:
                var = 0.0
            result[key] = {'mean': mean, 'variance': var}
    return result

def separate_mean_var(node, mean_node, var_node):
    for key, value in node.items():
        if isinstance(value, dict) and 'mean' not in value:
            mean_node[key] = {}
            var_node[key] = {}
            separate_mean_var(value, mean_node[key], var_node[key])
        else:
            mean_node[key] = value['mean']
            var_node[key] = value['variance']

def consolidation(data):
    output = {}
    
    for dname in data :
        entries = data.get(dname, [])
        collector = {}
        
        for entry in entries:
            metrics = entry.get('metrics', {})
            collect_metrics(collector, metrics)
        
        processed = process_collector(collector)
        
        output[dname] = {
            'mean': {},
            'variance': {}
        }
        
        separate_mean_var(processed, output[dname]['mean'], output[dname]['variance'])
    
    return output

if __name__ == '__consolidation_metrics__':
    consolidation()