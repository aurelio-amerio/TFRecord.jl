# using Pkg
# Pkg.activate(".")
using TFRecord

n = 10
f1 = rand(Bool, n)
f2 = rand(1:5, n)
f3 = rand(("cat", "dog", "chicken", "horse", "goat"), n)
f4 = rand(Float32, n)

TFRecord.write(
    "example.tfrecord",
    (
        Dict(
            "feature1" => f1[i],
            "feature2" => f2[i],
            "feature3" => f3[i],
            "feature4" => f4[i],
        )
        for i in 1:2
    )
)

for example in TFRecord.read("example.tfrecord")
    println(example)
end

read("example.tfrecord", String)
#%%
example_dict = Dict("feature1"=>420, "features2"=>314)
example = convert(TFRecord.Example, example_dict)

buff = IOBuffer()
e = ProtoEncoder(buff)
encode(e, example)

# data_crc = mask(crc32c(seekstart(buff)))
data = take!(seekstart(buff))
write("test.bin", data)
String(data)
#%%
ex = open("testpy.bin", "r") do io
    d = ProtoDecoder(io)
    decode(d, TFRecord.Example)
end

println("ex: ", ex)
println()
println("example: ", ex)
ex.features      #
example.features #


ex