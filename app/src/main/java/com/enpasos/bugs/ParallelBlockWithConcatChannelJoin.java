package com.enpasos.bugs;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.nn.ParallelBlock;

import java.util.List;
import java.util.stream.Collectors;

@SuppressWarnings("all")
public class ParallelBlockWithConcatChannelJoin extends ParallelBlock {


    public ParallelBlockWithConcatChannelJoin(List<Block> blocks) {
        super(list -> {
            List<NDArray> concatenatedList =
                list.stream().map(NDList::head).collect(Collectors.toList());
            return new NDList(NDArrays.concat(new NDList(concatenatedList), 1));
        }, blocks);
    }


}
