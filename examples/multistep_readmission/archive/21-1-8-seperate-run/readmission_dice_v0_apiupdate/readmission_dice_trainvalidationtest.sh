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
    esac
done

# echo "valueinput: ${valinput}";

echo "./bin/desc -inputfiles ${valinput} -outputfolder ${valoutput}"
./bin/desc -inputfiles ${valinput} -outputfolder ${valoutput}
echo "./bin/completer -inputfiles ${valinput} -inputfolder ${valoutput} -outputfolder ${valoutput} -visfolder ${valvis}"
./bin/completer -inputfiles ${valinput} -inputfolder ${valoutput} -outputfolder ${valoutput} -visfolder ${valvis}
