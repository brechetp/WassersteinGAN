__doc__ = '''utilities for the sqlite3
from http://www.sqlitetutorial.net/sqlite-python/ '''

import sqlite3
from sqlite3 import Error
from IPython import embed
import os
import pdb

TR_TABLE = {int: 'integer', str: 'varchar', float: 'real', list: 'varchar', bool: 'integer', None: 'varchar'}

def sqlf_opt_type(opt):
    '''format from a namespace to sqlite create a table'''
    res = 'id INTEGER PRIMARY KEY,'
    for i, kw in enumerate(opt._get_kwargs()):
        # if i <= 5:
            # continue
        res += '{} {},'.format(kw[0], TR_TABLE.get(type(kw[1]), 'varchar').upper())
    res = res[:-2]
    return res

def sqlf_opt(opt):
    '''format from a option objectto sqlite create a table'''

    field = ''
    placeholder = ''
    val = tuple()
    for i, kw in enumerate(opt._get_kwargs()):
        # if i <= 5:
            # continue
        val_str = str(tr(kw[1]))
        if val_str == '':
            continue
        field += '{},'.format(kw[0])
        placeholder += '{},'.format(val_str)
        val += (tr(kw[1]),)
    field = field[:-1]
    placeholder = placeholder[:-1]
    return field, placeholder, val

def strf_create_table(opt):

    dbname = opt.__dict__.get('dbname', 'runs')
    res = '''CREATE TABLE IF NOT EXISTS {} (
    {});'''.format(dbname, sqlf_opt_type(opt))
    return res

def strf_insert_into(opt):

    dbname = opt.__dict__.get('dbname', 'runs')
    fields, placeholder, val = sqlf_opt(opt)
    res = '''INSERT INTO {}({})
    VALUES({});'''.format(dbname, fields, placeholder)
    return res, val

def create_table_opt(opt):
    db_file = os.path.join(opt.outf, 'runs.db')
    conn = create_connection(db_file)
    sql_str = strf_create_table(opt)
    create_table(conn, sql_str)
    return conn


def tr(arg):
    out = arg
    if arg is None:
        out = ''
    elif arg == '':
        out = arg
    elif type(arg) is bool:
        out = int(arg)
    elif type(arg) is str or not isinstance(arg, (int, float)):
        out = "'{}'".format(arg)
    return out

def log_opt(conn, opt):

    sql, opt_val = strf_insert_into(opt)
    cur = conn.cursor()
    try:
        cur.execute(sql)
    except Error as e:
        print(e)
        pdb.set_trace()
    return cur.lastrowid



def create_table(conn, create_table_sql):
    '''Create a sql table'''
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def create_connection(db_file):

    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None

def main():
    embed()

if __name__ == '__main__':
    main()

