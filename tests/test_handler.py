import os
import json
import unittest
import pathlib
import subprocess

class TestOptimizerGenPlan(unittest.TestCase):
    
    def test_dummy_ftr1(self):
        sp = subprocess.Popen("python {}".format(os.path.join(pathlib.Path(__file__).parent.absolute(), 'dummy_ftr.py')), shell=True, stdout=subprocess.PIPE)
        sp_return = str(sp.stdout.read())
        sp_return = sp_return.split('\\n')

        training_plan = json.loads([l for l in sp_return if 'Model training plan:' in l][0].split('Model training plan:')[-1])
        assert training_plan["0_1_4_5"]["models"] == ["0", "1", "4", "5"]
        assert training_plan["0_1_4_5"]["output_layers"] == ["0/classifier", "1/classifier", "4/classifier", "5/classifier"]
        assert training_plan["0_1_4_5"]["flops_consumption"] == 2012681400000.0
        assert training_plan["0_1_4_5"]["compute_flops"] == 840806400000.0
        assert training_plan["0_1_4_5"]["load_flops"] == 1171875000000.0
        assert training_plan["0_1_4_5"]["memory_consumption"] == 1.032203733921051
        assert training_plan["0_1_4_5"]["used_materialized_layers"] == []
        assert training_plan["0_1_4_5"]["batch_size"] == 16
        assert training_plan["0_1_4_5"]["num_epochs"] == [5, 5, 5, 5]
        assert training_plan["0_1_4_5"]["flops_reduction"] == 1276783800000.0
        assert training_plan["0_1_4_5"]["merged_models"] == ["0_1", "4_5"]

        assert training_plan["2_3_6_7"]["models"] == ["2", "3", "6", "7"]
        assert training_plan["2_3_6_7"]["output_layers"] == ["2/classifier", "3/classifier", "6/classifier", "7/classifier"]
        assert training_plan["2_3_6_7"]["flops_consumption"] == 2012681400000.0
        assert training_plan["2_3_6_7"]["compute_flops"] == 840806400000.0
        assert training_plan["2_3_6_7"]["load_flops"] == 1171875000000.0
        assert training_plan["2_3_6_7"]["memory_consumption"] == 1.033119261264801
        assert training_plan["2_3_6_7"]["used_materialized_layers"] == []
        assert training_plan["2_3_6_7"]["batch_size"] == 32
        assert training_plan["2_3_6_7"]["num_epochs"] == [5, 5, 5, 5]
        assert training_plan["2_3_6_7"]["flops_reduction"] == 1276783800000.0
        assert training_plan["2_3_6_7"]["merged_models"] == ["2_3", "6_7"]

        assert float([l for l in sp_return if 'Current Practice FLOPs:' in l][0].split(':')[-1]) == 504176640.0
        assert float([l for l in sp_return if 'Optimal FLOPs:' in l][0].split(':')[-1]) == 252395520.0
        assert float([l for l in sp_return if 'Theoretical Speedup:' in l][0].split(':')[-1]) == 1.9975657254138266
        # print(sp_return)

    def test_dummy_ftr2(self):
        sp = subprocess.Popen("NAUTILUS_VERY_LARGE_VALUE=1e10 python {}".format(os.path.join(pathlib.Path(__file__).parent.absolute(), 'dummy_ftr.py --disk-throughput 99999999999')), shell=True, stdout=subprocess.PIPE)
        sp_return = str(sp.stdout.read())
        sp_return = sp_return.split('\\n')

        training_plan = json.loads([l for l in sp_return if 'Model training plan:' in l][0].split('Model training plan:')[-1])
        assert training_plan["0_1_4_5"]["models"] == ["0", "1", "4", "5"]
        assert training_plan["0_1_4_5"]["output_layers"] == ["0/classifier", "1/classifier", "4/classifier", "5/classifier"]
        assert training_plan["0_1_4_5"]["flops_consumption"] == 630988811718.75
        assert training_plan["0_1_4_5"]["compute_flops"] == 630988800000.0
        assert training_plan["0_1_4_5"]["load_flops"] == 11718.750000117187
        assert training_plan["0_1_4_5"]["memory_consumption"] == 1.0243835896253586
        assert training_plan["0_1_4_5"]["used_materialized_layers"] == ['layer_2', 'layer_4']
        assert training_plan["0_1_4_5"]["batch_size"] == 16
        assert training_plan["0_1_4_5"]["num_epochs"] == [5, 5, 5, 5]
        assert training_plan["0_1_4_5"]["flops_reduction"] == 0.0
        assert training_plan["0_1_4_5"]["merged_models"] == ["0_1", "4_5"]

        assert training_plan["2_3_6_7"]["models"] == ["2", "3", "6", "7"]
        assert training_plan["2_3_6_7"]["output_layers"] == ["2/classifier", "3/classifier", "6/classifier", "7/classifier"]
        assert training_plan["2_3_6_7"]["flops_consumption"] == 630988811718.75
        assert training_plan["2_3_6_7"]["compute_flops"] == 630988800000.0
        assert training_plan["2_3_6_7"]["load_flops"] == 11718.750000117187
        assert training_plan["2_3_6_7"]["memory_consumption"] == 1.0252991169691086
        assert training_plan["2_3_6_7"]["used_materialized_layers"] == ['layer_2', 'layer_4']
        assert training_plan["2_3_6_7"]["batch_size"] == 32
        assert training_plan["2_3_6_7"]["num_epochs"] == [5, 5, 5, 5]
        assert training_plan["2_3_6_7"]["flops_reduction"] == 0.0
        assert training_plan["2_3_6_7"]["merged_models"] == ["2_3", "6_7"]

        assert float([l for l in sp_return if 'Current Practice FLOPs:' in l][0].split(':')[-1]) == 504176640.0
        assert float([l for l in sp_return if 'Optimal FLOPs:' in l][0].split(':')[-1]) == 252395520.0
        assert float([l for l in sp_return if 'Theoretical Speedup:' in l][0].split(':')[-1]) == 1.9975657254138266

if __name__ == "__main__":
    unittest.main()