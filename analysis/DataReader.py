import argparse
import pickle

from utils import WORLD_REGIONS, SECTOR_GROUPS, get_regions_dict, get_sectors_dict
from dataformat import read_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('infile', type=str, help='Input file name.')
    parser.add_argument('outfile', type=str, help='Output file name.')
    parser.add_argument('--variable', type=str, default='all', help='Variable to read. If not specified, standard '
                                                                    'set of variables will beused.')
    parser.add_argument('--region', type=str, default='all', help='Region to read. If not specified, standard set'
                                                                  'of regions will beused.')
    parser.add_argument('--sector', type=str, default='all', help='Sector to read. If not specified, standard set'
                                                                  'of sectors will beused.')
    parser.add_argument('--time_frame', type=int, default=-1, help='')
    pars = vars(parser.parse_args())

    input_file = pars['infile']
    output_file = pars['outfile']
    time_frame = pars['time_frame']
    if pars['variable'] == 'all':
        variables = ['consumption', 'consumption_price', 'consumption_value', 'demand', 'demand_price', 'demand_value',
                     'production', 'production_price', 'production_value', 'storage', 'storage_price', 'storage_value',
                     'total_loss', 'total_loss_price', 'total_loss_value', 'total_value_loss', 'offer_price',
                     'expected_offer_price', 'expected_production', 'expected_production_price',
                     'expected_production_value', 'communicated_possible_production',
                     'communicated_possible_production_price', 'communicated_possible_production_value',
                     'unit_production_costs', 'total_production_costs', 'total_revenue', 'direct_loss',
                     'direct_loss_price', 'direct_loss_value', 'forcing', 'incoming_demand', 'incoming_demand_price',
                     'incoming_demand_value', 'production_capacity', 'desired_production_capacity',
                     'possible_production_capacity']
    elif pars['variable'] == 'set_sandy':
        variables = ['production', 'production_value', 'incoming_demand', 'incoming_demand_value', 'consumption',
                     'consumption_value', 'forcing']
    else:
        variables = pars['variable'].split('+')

    if pars['region'] == 'all':
        regions = set(list(WORLD_REGIONS.keys()) + WORLD_REGIONS['WORLD'])
    elif pars['region'] == 'set_sandy':
        regions = set(list(WORLD_REGIONS['USA']) + ['DEU', 'CHN', 'EUR', 'WORLD', 'ROW', 'USA_REST_SANDY'])
    else:
        regions = pars['region'].split('+')

    if pars['sector'] == 'all':
        sectors = [i for i in SECTOR_GROUPS['ALLSECTORS']] + ['PRIVSECTORS']
    else:
        sectors = pars['sector'].split('+')

    region_args = get_regions_dict(regions)
    sector_args = get_sectors_dict(sectors)

    data = read_file(input_file, variables, region_args, sector_args, time_frame)
    try:
        pickle.dump(data, open(output_file, 'wb'))
        print("Saved as {}".format(output_file))
    except Exception as e:
        print(e)
