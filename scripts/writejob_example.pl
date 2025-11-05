# Inputs
if ((scalar @ARGV) < 4) {
    print "Usage: perl writejobs.pl <account> <config file> <output tag> <script location>\n";
    die;
}
($useAccount, $cfg, $tag, $job) = @ARGV;

# Stuff below here you shouldn't need to modify just to do another run.
# You *will* need to modify it if you move to a different platform: OSC
# uses $TMPDIR to tell us what directory to use for on-node temporary
# storage, if your platform uses something else you'll need to adjust
# accordingly.

print "Using account -->         $useAccount\n";
print "Config file -->           $cfg\n";
print "Output logs -->           $tag-*\n";
print "Job script location -->   $job-*\n";

# Python script to use
$py  = "import sys\n";
$py .= "import os\n";
$py .= "import numpy as np\n";
$py .= "from pyimcom.config import Config, Settings\n";
$py .= "from pyimcom.coadd import Block\n";
$py .= "from pyimcom.truthcats import gen_truthcats_from_cfg\n";
$py .= "cfg = Config(sys.argv[1])\n";
$py .= "if len(sys.argv)==2:\n";
$py .= "    print(cfg.nblock)\n";
$py .= "    print(cfg.outstem)\n";
$py .= "    exit()\n";
$py .= "if len(sys.argv)>3:\n";
$py .= "    if sys.argv[3]=='draw':\n";
$py .= "        cfg.stoptile=4\n";
$py .= "    if sys.argv[3]=='reduce':\n";
$py .= "        cfg.instamp_pad = 0.7 * Settings.arcsec\n";       # <-- can change INPAD for the first iteration here
$py .= "cfg.tempfile = os.getenv('TMPDIR') + '/temp'\n";          # <-- can change $TMPDIR here
$py .= "cfg()\n";
$py .= "print(cfg.to_file(None))\n";
$py .= "block = Block(cfg=cfg, this_sub=int(sys.argv[2]))\n";
$py .= "if int(sys.argv[2])==0: gen_truthcats_from_cfg(cfg)\n";

open(OUT, ">", "$job\_scr.py");
print OUT $py;
close OUT;

# script information
$ngrp = 4;
@data = split ' ', `python $job\_scr.py $cfg`;
$nblock = int($data[0]);
$outstem = $data[1];
$pergrp = int($nblock**2/$ngrp);
if ($nblock**2 % $ngrp > 0) {
    print STDERR "Error: incorrect sizes. $ngrp groups in $nblock x $nblock doesn't work.\n";
    die;
}
$gmax = $pergrp - 1;
$nmax = $nblock**2 - 1;
$nblock2 = $nblock**2;

print "Output stem -->           $outstem*\n";
print "Mosaic size -->           $nblock x $nblock blocks\n";

$njob = 0;

# PSF splitting
$scriptHead = "#!/bin/bash\n#SBATCH --job-name=pyimcom\n#SBATCH --account=$useAccount\n";
$script0  = "$scriptHead#SBATCH --time=24:00:00\n#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4\n";
$script0 .= "cd \$SLURM_SUBMIT_DIR\n";
$script0 .= "python3 -m pyimcom.splitpsf.splitpsf $cfg > $tag-S$njob.txt\n";
open(OUT, ">", "$job-$njob.job");
print OUT $script0;
close OUT;

# Script to build the input layers
$njob++;
$script1  = "$scriptHead#SBATCH --array=0-2\n#SBATCH --time=24:00:00\n#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2\n";
$script1 .= "cd \$SLURM_SUBMIT_DIR\n";
$script1 .= "STARTBLOCK=\$(($pergrp*SLURM_ARRAY_TASK_ID))\n";
$script1 .= "for i in {0..$gmax}; do\n";
$script1 .= "    BLOCK=\$((STARTBLOCK+i))\n";
$script1 .= "    python $job\_scr.py $cfg \$BLOCK draw > $tag-S$njob-\$BLOCK.txt\n";
$script1 .= "done\n";
open(OUT, ">", "$job-$njob.job");
print OUT $script1;
close OUT;

# First run of pyimcom
$njob++;
$script2  = "$scriptHead#SBATCH --array=0-$nmax\n#SBATCH --time=24:00:00\n#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2\n";
$script2 .= "cd \$SLURM_SUBMIT_DIR\n";
$script2 .= "python $job\_scr.py $cfg \$SLURM_ARRAY_TASK_ID reduce > $tag-S$njob-\$SLURM_ARRAY_TASK_ID.txt\n";
open(OUT, ">", "$job-$njob.job");
print OUT $script2;
close OUT;

# Imsubtract
$njob++;
$script3  = "$scriptHead#SBATCH --array=1-18\n#SBATCH --time=48:00:00\n#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=6\n";
$script3 .= "cd \$SLURM_SUBMIT_DIR\n";
$script3 .= "python3 -m pyimcom.splitpsf.imsubtract $cfg \$SLURM_ARRAY_TASK_ID > $tag-S$njob-\$SLURM_ARRAY_TASK_ID.txt\n";
open(OUT, ">", "$job-$njob.job");
print OUT $script3;
close OUT;

# Pass back
$njob++;
$script4  = "$scriptHead#SBATCH --time=8:00:00\n#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4\n";
$script4 .= "cd \$SLURM_SUBMIT_DIR\n";
$script4 .= "python3 -m pyimcom.splitpsf.update_cube $cfg > $tag-S$njob.txt\n";
open(OUT, ">", "$job-$njob.job");
print OUT $script4;
close OUT;

# Final run of pyimcom
$njob++;
$script5  = "$scriptHead#SBATCH --array=0-$nmax\n#SBATCH --time=24:00:00\n#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2\n";
$script5 .= "cd \$SLURM_SUBMIT_DIR\n";
$script5 .= "python $job\_scr.py $cfg \$SLURM_ARRAY_TASK_ID > $tag-S$njob-\$SLURM_ARRAY_TASK_ID.txt\n";
open(OUT, ">", "$job-$njob.job");
print OUT $script5;
close OUT;

# Compression
open(COUT, ">", "$job\_cprs.py");
print COUT "import sys\n";
print COUT "from pyimcom.compress.compressutils import CompressedOutput\n";
print COUT "from pyimcom.config import Config\n";
print COUT "cfg = Config(sys.argv[1])\n";
print COUT "for i in range($nblock2):\n";
print COUT "    ibx = i%$nblock\n";
print COUT "    iby = i//$nblock\n";
print COUT "    fname = cfg.outstem + f'_{ibx:02d}_{iby:02d}.fits'\n";
print COUT "    fout = cfg.outstem + f'_{ibx:02d}_{iby:02d}.cpr.fits.gz'\n";
print COUT "    print(fname, '-->', fout); sys.stdout.flush()\n";
print COUT "\n";
print COUT "    with CompressedOutput(fname) as f:\n";
print COUT "        for j in range(1,len(f.cfg.extrainput)):\n";
print COUT "            if f.cfg.extrainput[j][:6].lower()=='gsstar' or f.cfg.extrainput[j][:5].lower()=='cstar'\n";
print COUT "                    or f.cfg.extrainput[j][:8].lower()=='gstrstar' or f.cfg.extrainput[j][:8].lower()=='gsfdstar':\n";
print COUT "                f.compress_layer(j, scheme='I24B', pars={'VMIN': -1./64., 'VMAX': 7./64., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})\n";
print COUT "            if f.cfg.extrainput[j][:5].lower()=='nstar':\n";
print COUT "                f.compress_layer(j, scheme='I24B', pars={'VMIN': -1500., 'VMAX': 10500., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})\n";
print COUT "            if f.cfg.extrainput[j][:5].lower()=='gsext':\n";
print COUT "                f.compress_layer(j, scheme='I24B', pars={'VMIN': -1./64., 'VMAX': 7./64., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})\n";
print COUT "            if f.cfg.extrainput[j][:10].lower()=='whitenoise':\n";
print COUT "                f.compress_layer(j, scheme='I24B', pars={'VMIN': -8, 'VMAX': 8, 'BITKEEP': 14, 'DIFF': True, 'SOFTBIAS': -1})\n";
print COUT "            if f.cfg.extrainput[j][:7].lower()=='1fnoise':\n";
print COUT "                f.compress_layer(j, scheme='I24B', pars={'VMIN': -32, 'VMAX': 32, 'BITKEEP': 14, 'DIFF': True, 'SOFTBIAS': -1})\n";
print COUT "            if f.cfg.extrainput[j][:6].lower()=='noise,':\n";
print COUT "                f.compress_layer(j, scheme='I24B', pars={'VMIN': -0.125, 'VMAX': 0.125, 'BITKEEP': 14, 'DIFF': True, 'SOFTBIAS': -1})\n";
print COUT "        f.to_file(fout, overwrite=True)\n";
close COUT;
$njob++;
$script6  = "$scriptHead#SBATCH --time=8:00:00\n#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4\n";
$script6 .= "cd \$SLURM_SUBMIT_DIR\n";
$script6 .= "python $job\_compress.py $cfg > $tag-S$njob.txt\n";
open(OUT, ">", "$job-$njob.job");
print OUT $script6;
close OUT;

# Diagnostic report
$njob++;
$script7  = "$scriptHead#SBATCH --time=8:00:00\n#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4\n";
$script7 .= "cd \$SLURM_SUBMIT_DIR\n";
$script7 .= "python3 -m pyimcom.diagnostics.run $outstem\_00\_00.cpr.fits.gz $tag\_report > $tag-S$njob.txt";
open(OUT, ">", "$job-$njob.job");
print OUT $script7;
close OUT;

# Print the possible jobs
for $i (0..$njob) {
    print "=== Job $i ===\n";
    system "cat $job-$i.job";
    print "\n";
}

print "\nSubmit jobs? [Y/N]: ";
$response = <STDIN>;
chomp $response;

if (($response eq 'Y') or ($response eq 'y')) {
    print "Setting up jobs ...\n";
    $cmd = "sbatch $job-0.job";
    print "\#0: $cmd\n";
    $id = `$cmd`;
    $id = int((split ' ', $id)[-1]);
    print "$Job " . (sprintf "%2d", 0) . ": id = $id\n";
    for $i (1..$njob) {
        $cmd = "sbatch --dependency=afterok:$id $job-$i.job";
        print "\#$i: $cmd\n";
        $id = `$cmd`;
        $id = int((split ' ', $id)[-1]);
        print "$Job " . (sprintf "%2d", $i) . ": id = $id\n";
    }
}
else {
    print "Exiting ...\n";
}
