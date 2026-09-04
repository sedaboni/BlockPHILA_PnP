for sf in 1 2;

do
    python main.py --opt=DPIR --sf=$sf  --comment=DPIR
    python main.py --opt=GSPnP --sf=$sf  --comment=PGD

    for nblocks in 1 2 4;
    do
        python main.py --opt=GD_block --sf=$sf  --nblocks=$nblocks --comment=phila8
        python main.py --opt=iGD_block --sf=$sf  --nblocks=$nblocks --comment=phila7
        python main.py --opt=GD_block --sf=$sf  --nblocks=$nblocks --BB=geo --comment=phila6
        python main.py --opt=iGD_block --sf=$sf  --nblocks=$nblocks --BB=geo --comment=phila5
        if [ $nblocks = 1 ] 
        then
            python main.py --opt=VMILA --sf=$sf  --nblocks=$nblocks --comment=phila4
            python main.py --opt=PHILA --sf=$sf  --nblocks=$nblocks --comment=phila3
            python main.py --opt=VMILA --sf=$sf  --nblocks=$nblocks --BB=geo --comment=phila2
            python main.py --opt=PHILA --sf=$sf  --nblocks=$nblocks --BB=geo --comment=phila1
            python main.py --opt=VMILA_vm --sf=$sf  --nblocks=$nblocks --comment=phila4_vm --prox=dual
            python main.py --opt=PHILA_vm --sf=$sf  --nblocks=$nblocks --comment=phila3_vm --prox=dual
            python main.py --opt=VMILA_vm --sf=$sf  --nblocks=$nblocks --BB=geo_vm --comment=phila2_vm --prox=dual
            python main.py --opt=PHILA_vm --sf=$sf  --nblocks=$nblocks --BB=geo_vm --comment=phila1_vm --prox=dual
        else
            python main.py --opt=VMILA_block --sf=$sf  --nblocks=$nblocks --prox=dual --comment=phila4
            python main.py --opt=PHILA_block --sf=$sf  --nblocks=$nblocks --prox=dual --comment=phila3
            python main.py --opt=VMILA_block --sf=$sf  --nblocks=$nblocks --BB=geo --prox=dual --comment=phila2
            python main.py --opt=PHILA_block --sf=$sf  --nblocks=$nblocks --BB=geo --prox=dual --comment=phila1
            python main.py --opt=VMILA_block_vm --sf=$sf  --nblocks=$nblocks --comment=phila4_vm --prox=dual
            python main.py --opt=PHILA_block_vm --sf=$sf  --nblocks=$nblocks --comment=phila3_vm --prox=dual
            python main.py --opt=VMILA_block_vm --sf=$sf  --nblocks=$nblocks --BB=geo_vm --comment=phila2_vm --prox=dual
            python main.py --opt=PHILA_block_vm --sf=$sf  --nblocks=$nblocks --BB=geo_vm --comment=phila1_vm --prox=dual
        fi

    done
done