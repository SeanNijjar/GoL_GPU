
#test.pl
use 5.012;
use strict;
use warnings;
use Getopt::Long;

my $usage = "-tests <testDir> -ref <referenceProg> -testprog <gpuversion> -iters <#iters>\n";

my $testDir       = "";
my $referenceProg = "";
my $progUnderTest = "";
my $iters         = "";

GetOptions ("tests=s"    => \$testDir,
            "ref=s"      => \$referenceProg,
			"testprog=s" => \$progUnderTest,
			"iters=s"    => \$iters);
			
if("" eq $testDir or
   "" eq $referenceProg or
   "" eq $progUnderTest or 
   "" eq $iters) {
	die("bad args: $usage\n");
}

################################################
## Get the test files 
################################################
opendir(my $testDirHandle, $testDir) or die("Couldn't open tests\n");
my @testFiles = readdir $testDirHandle;
closedir $testDirHandle;

### Make
`make`;

### Run tests
my $tempFile = "temp.txt";
foreach my $testFile (@testFiles) {
	my ($rows, $cols) = WriteToFlatFile("$testDir\/$testFile", $tempFile);
	my $testProgCmd = "$progUnderTest $cols $rows $iters $tempFile";
	my $refProgCmd = "$referenceProg $cols $rows $iters $tempFile ";
	print"$testProgCmd \n";
	print"$refProgCmd \n";
	my @result = `$testProgCmd`;
	my @refResult = `$refProgCmd`;
	my $MSG = "";
	my $bResultsMatch = ResultsMatch(\@result, \@refResult, $cols);
	print("result: $bResultsMatch\n");
	if(1 == $bResultsMatch) {
		$MSG = "SUCCESS";
	} else {
		$MSG = "FAILURE";
	}
	print("$MSG: $testFile\n");
}

sub ResultsMatch {
	my $testResultsRef = $_[0];
	my $refResultsRef  = $_[1];
	my $width = $_[2];
	my @testResults = @$testResultsRef;
	my @refResults = @$refResultsRef;

	my $size = @testResults;
#	if($size != @refResults) { die("Output gridsize mismatch\n"); }
	for(my $i = 0; $i < $size; $i = $i + 1) {
		if(substr($testResults[$i], 0, $width) ne substr($refResults[$i], 0, $width)) {
			print "FAILURE\n";
			print "TEST RESULTS A:\n @testResults\n\n";
			print "TEST RESULTS B:\n @refResults\n\n";
			return 0;
		}
	}
	return 1;
}

sub WriteToFlatFile{
	my $testFile = $_[0];
	my $tempFile = $_[1];
	print("testfile: $testFile\n");
	
	### Read the test file
	my $flatFile = "";
	open(my $TEST_FH, "<", $testFile);
	my @gridLines = <$TEST_FH>;
	close($testFile);
	
	my $rows = @gridLines;
	my $cols = length($gridLines[0]) - 1;
	
	print("Reading in level:\n");
	foreach my $row(@gridLines) {
		$row =~ s/\n//;
		$row =~ s/[#Xx1]/X/gi;
		$row =~ s/[_ 0]/ /gi;
		print("$row\n");
		$flatFile = "$flatFile$row";
	}
	print("\n");
	
	open(my $FLAT_FH, ">", $tempFile) or die;
	print $FLAT_FH $flatFile;
	close($FLAT_FH);
	
	print("flat file: $rows x $cols\n");
	return ($rows, $cols);
}
