optspec=":-:"
while getopts "$optspec" optchar; do
    # echo "OPTARG : ${OPTARG}"
    case "${OPTARG}" in
        input)
            valinput="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            # echo "Parsing option: '--${OPTARG}', value: '${valinput}'" >&2;
            ;;
        output)
            valoutput="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            # echo "Parsing option: '--${OPTARG}', value: '${valoutput}'" >&2;
            ;;
        vis)
            valvis="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            # echo "Parsing option: '--${OPTARG}', value: '${valvis}'" >&2;
            ;;
        sampleid)
            valsampleid="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            # echo "Parsing option: '--${OPTARG}', value: '${valsampleid}'" >&2;
            ;;
        trainratio)
            valtrainratio="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            # echo "Parsing option: '--${OPTARG}', value: '${valtrainratio}'" >&2;
            ;;
        validationratio)
            valvalidationratio="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            # echo "Parsing option: '--${OPTARG}', value: '${valvalidationratio}'" >&2;
            ;;
        testratio)
            valtestratio="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            # echo "Parsing option: '--${OPTARG}', value: '${valtestratio}'" >&2;
            ;;
    esac
done


# echo "valueinput: ${valinput}";
# echo "valueoutput: ${valoutput}";
# echo "valuevis: ${valvis}";
# echo "valuesampleid: ${valsampleid}";
# echo "valtrainratio: ${valtrainratio}"
# echo "valvalidationratio: ${valvalidationratio}"
# echo "valtestratio: ${valtestratio}"


echo "python ./CNN-readmission-trainvalidationtest.py -inputfolder ${valinput} -outputfolder ${valoutput} -visfolder ${valvis} -trainratio ${valtrainratio} -validationratio ${valvalidationratio} -testratio ${valtestratio}"
python ./CNN-readmission-trainvalidationtest.py -inputfolder ${valinput} -outputfolder ${valoutput} -visfolder ${valvis} -trainratio ${valtrainratio} -validationratio ${valvalidationratio} -testratio ${valtestratio}
